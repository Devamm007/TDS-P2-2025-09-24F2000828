from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
import asyncio

from dotenv import load_dotenv
from os import getenv
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import re
import pytz
from datetime import datetime

import uvicorn

IST = pytz.timezone('Asia/Kolkata')
app = FastAPI()

# Configuration
templates_dir = Path(__file__).parent / "templates"
app.mount("/templates", StaticFiles(directory=str(templates_dir)), name="templates")
load_dotenv()
app.state.SECRET = getenv("SECRET")
app.state.LLM_API_KEY = getenv("LLM_API_KEY")
app.state.EMAIL = getenv("EMAIL")
app.state.LLM_BASE_URL = getenv("LLM_BASE_URL")
app.state.LLM_MODEL = getenv("LLM_MODEL")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the index.html page"""
    with open(templates_dir / "index.html") as f:
        return HTMLResponse(content=f.read())

def validate_secret(secret: str) -> bool:
    return secret == app.state.SECRET

def validate_email(email: str) -> bool:
    return email == app.state.EMAIL

def validate_url(url: str) -> bool:
    ''' validate url return True or string with error message '''
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        if result.scheme != "https" or not result.netloc:
            return "Invalid URL format"
        return True
    except Exception:
        return "Invalid URL format"

async def solve_with_llm(context: str, previous_reason: str = None) -> str:
    """
    Sends scraped content to LLM to extract the question and find the answer.
    """
    system_prompt = (
        "You are an automated agent solving a Capture The Flag (CTF) quiz. "
        "Your goal is to extract the question from the provided webpage content "
        "and provide the specific answer. "
        "Return ONLY the answer. If it is a number, return just the digits. "
        "If it is a word, return just the word. "
        "If it is a phrase, return just the phrase without any quotes. "
        "If it is data, return JSON format with keys and values (values maybe data URI/attachment)."
    )
    
    user_prompt = f"Webpage Content:\n{context}\n"
    
    if previous_reason:
        user_prompt += f"\n\nPREVIOUS ATTEMPT FAILED. Reason: {previous_reason}. Try a different approach."

    payload = {
        "model": app.state.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {app.state.LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=150) as client:
            response = await client.post(
                f"{app.state.LLM_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=150
            )
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        answer = answer.replace("`", "").replace("'", "").replace('"', "")
        return answer
    except Exception as e:
        print(f"--- LLM Error: {e} ---")
        return "0" # Fallback

async def process_task(url: str, reason: str = None) -> dict:
    """
    1. Scrape the data from URL or nested URLs
    2. Extract relevant content from various file types
    (Usually headless browser + file parsers)
    (html, script, css files of url and nested urls,
    audio(.mp3, .wav, .flac, .aac, .ogg, .m4a, .alac, and .wma),
    image(.jpg, .jpeg, .png, .gif, .webp, .svg, .avif),
    video(.mkv, .mp4, .mov, .avi, .wmv, .flv, .webm, .mpeg),
    csv, xlxs, docx, text, pdf(text, ocr, tables), zip files, tar files etc.)
    3. Understand the question/task
    4. Process it
    5. Return the answer
    """

    print(f"--- [{datetime.now(IST)}] Scraping: {url} ---")

    scraped_content = ""
    try:
        browser_config = BrowserConfig(verbose=False, headless=True)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if result.success:
                # We prefer Markdown for LLMs, but fallback to cleaned HTML
                scraped_content = result.markdown or result.cleaned_html
                print(f"--- Scraped {len(scraped_content)} chars. ---")
            else:
                print(f"--- Scraping failed: {result.error_message} ---")
                return None
    except Exception as e:
        print(f"--- Crawler Exception: {e} ---")
        return None

    # 2. Solve
    print(f"--- Analyzing with LLM... ---")
    answer = await solve_with_llm(scraped_content, reason)
    print(f"--- Calculated Answer: {answer} ---")

    return {
            "email": app.state.EMAIL,
            "secret": app.state.SECRET,
            "url": url,
            "answer": answer
        }

async def worker(url: str):
    """
    Worker function to handle the task processing and submission loop.
    """

    max_iterations = 400
    iteration_count = 0
    questions_data = []
    questions_data.append({
        "url": url,
        "question_number": 1 + len(questions_data),
        "status": False,
        "started_at": datetime.now(IST),
        "completed_at": None,
        "answer": None
    })
    print("=== START EVALUATION ===\n")
    domain_match = re.search(r'^(?:http|ftp)s?://([^/]+)', url)
    domain = domain_match.group(1) if domain_match else 'default'

    # initial processing
    payload = await process_task(url)
    while True:
        # subsequent processing based on response  
        if not payload:
            print(f"--- Failed to generate payload for submission. Ending evaluation. ---\n")
            print("=== END EVALUATION DUE TO PAYLOAD ERROR ===\n")
            return None        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(f"https://{domain}/submit", json=payload)
        except Exception as e:
            print(f"--- NETWORK ERROR ---\nTYPE: {type(e)}\nDETAILS: {str(e)}\n--- END NETWORK ERROR ---\n")
            print(f"=== END EVALUATION DUE TO NETWORK ERROR ===")
            return None
        # error, server error status codes handling
        if response.status_code in [400, 403, 404, 500, 502, 503, 504]:
            print(f"--- ERROR ---\nTYPE: HTTP {response.status_code}\nDETAILS: {response.text}\n--- END ERROR ---\n")
            print(f"=== END EVALUATION DUE TO ERROR ===")
            return None
        
        # handle is response not json convertable
        try:
            response.json()
        except Exception as e:
            print(f"--- ERROR ---\nTYPE: Invalid JSON Response\nDETAILS: {response.text}\n--- END ERROR ---\n")
            print(f"=== END EVALUATION DUE TO INVALID JSON RESPONSE ===")
            return None
        data = response.json()
        last_question = questions_data[-1]
        questions_data[-1]["answer"] = payload.get("answer", None)

        print(f"--- Answer submitted successfully to {domain}/submit ---\n")
        print(f"--- Submission Response ---\n{response.json()}\n--- End Submission Response ---\n")

        if data.get("correct") and data.get("url"):
            if data.get("correct") is True:
                print(f"--- Answer correct. Proceeding to next URL: {data.get('url')} ---\n")
                # change the status of last question to True
                questions_data[-1]["status"] = True
                questions_data[-1]["completed_at"] = datetime.now(IST)
                questions_data.append({
                    "url": data.get('url', None),
                    "question_number": 1 + len(questions_data),
                    "status": False,
                    "started_at": datetime.now(IST),
                    "completed_at": None,
                    "answer": None
                })
                payload = await process_task(data.get("url"))
            else:
                last_question = questions_data[-1]
                time_diff = datetime.now(IST) - last_question["started_at"]
                if time_diff.total_seconds() > 60:
                    print(f"--- Time exceeded for {last_question['url']}. Proceeding to next URL: {data.get('url')} ---\n")
                    questions_data[-1]["status"] = False
                    questions_data[-1]["completed_at"] = datetime.now(IST)
                    questions_data.append({
                        "url": data.get('url', None),
                        "question_number": 1 + len(questions_data),
                        "status": False,
                        "started_at": datetime.now(IST),
                        "completed_at": None,
                        "answer": None
                    })
                    payload = await process_task(data.get("url"))
                else:
                    if data.get("reason"):
                        print(f"--- Answer incorrect. Reason: {data.get('reason')} ---\n")
                    payload = await process_task(last_question["url"], data.get("reason"))
        elif data.get("correct") and not data.get("url"):
            if data.get('correct') is False:
                last_question = questions_data[-1]
                time_diff = datetime.now(IST) - last_question["started_at"]
                if time_diff.total_seconds() > 150:
                    print(f"--- Time exceeded for {last_question['url']}. Ending evaluation. ---\n")
                    questions_data[-1]["status"] = False
                    questions_data[-1]["completed_at"] = datetime.now(IST)
                    print(f"--- QUESTIONS SUMMARY ---")
                    for q in questions_data:
                        print(f"Question {q['question_number']}: URL: {q['url']}, Status: {'Correct' if q['status'] else 'Incorrect'}, Started at: {q['started_at'].strftime('%Y-%m-%d %H:%M:%S')}, Completed at: {q['completed_at'].strftime('%Y-%m-%d %H:%M:%S') if q['completed_at'] else 'N/A'}")
                    print("=== END EVALUATION ===\n")
                    break
                else:
                    print(f"--- Answer incorrect. Reason: {data.get('reason')} ---\n")
                    questions_data[-1]["status"] = False
                    payload = await process_task(last_question["url"], data.get("reason"))
            else:
                print(f"--- Answer correct. Proceeding to next URL: {data.get('url')} ---\n")
                # change the status of last question to True
                questions_data[-1]["status"] = True
                questions_data[-1]["completed_at"] = datetime.now(IST)
                # log all questions data in logging format
                print(f"--- QUESTIONS SUMMARY ---")
                for q in questions_data:
                    print(f"Question {q['question_number']}: URL: {q['url']}, Status: {'Correct' if q['status'] else 'Incorrect'}, Started at: {q['started_at'].strftime('%Y-%m-%d %H:%M:%S')}, Completed at: {q['completed_at'].strftime('%Y-%m-%d %H:%M:%S') if q['completed_at'] else 'N/A'}")
                print("=== END EVALUATION ===\n")
                break
            
        iteration_count += 1
        await asyncio.sleep(1)  # brief pause to avoid overwhelming the server
        if iteration_count >= max_iterations:
            print(f"--- QUESTIONS SUMMARY ---")
            for q in questions_data:
                print(f"Question {q['question_number']}: URL: {q['url']}, Status: {'Correct' if q['status'] else 'Incorrect'}, Started at: {q['started_at'].strftime('%Y-%m-%d %H:%M:%S')}, Completed at: {q['completed_at'].strftime('%Y-%m-%d %H:%M:%S') if q['completed_at'] else 'N/A'}")
            print(f"--- Maximum iterations reached. Ending evaluation. ---\n")
            print("=== END EVALUATION ===\n")
            break
    return None            

@app.post("/handle_task")
async def handle_task(data: dict):
    """
    Endpoint to handle incoming tasks for processing.
    Validates the input and initiates background processing.
    """

    # validate secret, email and JSON validity
    if data.get('secret') and data.get('email') and data.get('url'):
        # validate secret
        if not validate_secret(data.get('secret', '')):
            print("--- Invalid secret detected. ---\n")
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        # validate email
        if not validate_email(data.get('email', '')):
            print("--- Invalid email detected. ---\n")
            raise HTTPException(status_code=400, detail="Invalid email")
        
        # validate url
        url_validation_result = validate_url(data.get("url", ""))
        if type(url_validation_result) != bool:
            print(f"--- URL validation failed: {url_validation_result} ---\n")
            raise HTTPException(status_code=400, detail=url_validation_result)
        
        print(f"--- Processing URL {data.get('url')} ---\n")

        asyncio.create_task(worker(data.get("url")))
    else:
        print("--- Invalid JSON structure detected. ---\n")
        raise HTTPException(status_code=400, detail="Invalid JSON structure")

    return {"status": "Secret validated. Task is being processed in the background."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)