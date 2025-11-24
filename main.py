# /// script
# requires python3 = ">=3.11"
# requirements = [
#     "fastapi>=0.95.2",
#     "uvicorn>=0.22.0",
#     "PyMuPDF>=1.22.5"
# ]
# ///

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests

from dotenv import load_dotenv
from os import getenv
from pathlib import Path

import base64
import re
from time import sleep
import pytz
from datetime import datetime

IST = pytz.timezone('Asia/Kolkata')
app = FastAPI()

# Configuration
templates_dir = Path(__file__).parent / "templates"
app.mount("/templates", StaticFiles(directory=str(templates_dir)), name="templates")
load_dotenv()
app.state.SECRET = getenv("SECRET")
app.state.LLM_API_KEY = getenv("LLM_API_KEY")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the index.html page"""
    with open(templates_dir / "index.html") as f:
        return HTMLResponse(content=f.read())

def validate_secret(secret: str) -> bool:
    return secret == app.state.SECRET

def validate_email(email: str) -> bool:
    return email == "24f2000828@ds.study.iitm.ac.in"

def validate_url(url: str) -> bool: 
    ''' validate url reutrn True or string with error message '''
    # Basic URL validation
    url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
        r'localhost|' # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|' # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)' # ...or ipv6
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if re.match(url_regex, url) is None:
        return f"Invalid URL format: {url}"

    # Check URL reachability
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code != 200:
            return f"URL is not reachable, status code: {response.status_code}"
    except requests.RequestException as e:
        return f"URL is not reachable: {url}\nException: ({str(e)})"
    return True

def process_task(url: str, reason: str = None) -> ...:
    '''Keep checking for incoming JSON body to API endpoint
    and run below while JSON body keeps coming with a URL'''
    try:
        pass
    except Exception as e:
        print(f"--- ERROR ---\nTYPE: {type(e)}\nDETAILS: {str(e)}\n--- END ERROR ---\n")
           
    return ...

# INPUT (REPEATED INTERACTIONS):
# {
#   "email": "your email", // Student email ID
#   "secret": "your secret", // Student-provided secret
#   "url": "https://example.com/quiz-834" // A unique task URL
#   // ... other fields
# }

# OUTPUT (FOR EACH INTERACTION):
# {
#   "email": "your email",
#   "secret": "your secret",
#   "url": "https://example.com/quiz-834",
#   "answer": 12345 // the correct answer
# }

# RESULT OF INTERACTION:
# {
#   "correct": true,
#   "url": "https://example.com/quiz-942",
#   "reason": null
#   // ... other fields
# }

# FORMAT FOR LAST INTERACTION (WHEN NO FURTHER QUESTIONS):
# {
#   "correct": false,
#   "reason": "The sum you provided is incorrect."
#   // maybe with no new url provided
# }

# Storing questions url, answer, status
questions_data = []

@app.post("/handle_task")
def handle_task(data: dict, background_tasks: BackgroundTasks):
    # validate email and secret if in JSON body
    ## for first interaction
    if data.get('secret') and data.get('email'):
        # validate secret
        if not validate_secret(data.get('secret', '')):
            print("--- Invalid secret detected. ---\n")
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        # validate email
        if not validate_email(data.get('email', '')):
            print("--- Invalid email detected. ---\n")
            raise HTTPException(status_code=400, detail="Invalid email")

        # start background task processing
        print("=== START EVALUATION ===\n")
        questions_data.append({
            "url": data.get('url', None),
            "question_number": 1 + len(questions_data),
            "status": False,
            "started_at": datetime.now(IST),
            "completed_at": None
        })

    ## for subsequent interactions
    if data.get("correct") and data.get("url"):
        if data.get("correct"):
            print(f"--- Answer correct. Proceeding to next URL: {data.get('url')} ---\n")
            # change the status of last question to True
            questions_data[-1]["status"] = True
            questions_data[-1]["completed_at"] = datetime.now(IST)
            questions_data.append({
                "url": data.get('url', None),
                "question_number": 1 + len(questions_data),
                "status": False,
                "started_at": datetime.now(IST),
                "completed_at": None
            })
            background_tasks.add_task(process_task, data.get("url"))
        else:
            last_question = questions_data[-1]
            time_diff = datetime.now(IST) - last_question["started_at"]
            if time_diff.total_seconds() > 60:
                print(f"--- Time exceeded for {last_question['url']}. Proceeding to next URL: {data.get('url')} ---\n")
                background_tasks.add_task(process_task, data.get("url"))
            else:
                if data.get("reason"):
                    print(f"--- Answer incorrect. Reason: {data.get('reason')} ---\n")
                background_tasks.add_task(process_task, last_question["url"], data.get("reason"))            
                
    ## for first interaction
    if data.get("url") and not data.get("correct"):
        url_validation_result = validate_url(data.get("url", ""))
        if type(url_validation_result) != bool:
            print(f"--- URL validation failed: {url_validation_result} ---\n")
            raise HTTPException(status_code=400, detail=url_validation_result)
        
        print(f"--- Processing URL {data.get('url')} ---\n")
        background_tasks.add_task(process_task, data.get("url"))
        
    ## for last interaction
    if data.get("correct") and not data.get("url"):
        if not data.get('correct'):
            last_question = questions_data[-1]
            time_diff = datetime.now(IST) - last_question["started_at"]
            if time_diff.total_seconds() > 120:
                print(f"--- Time exceeded for {last_question['url']}. Ending evaluation. ---\n")
            else:
                print(f"--- Answer incorrect. Reason: {data.get('reason')} ---\n")
                background_tasks.add_task(process_task, last_question["url"], data.get("reason"))
        else:
            print(f"--- Answer correct. Proceeding to next URL: {data.get('url')} ---\n")
            # change the status of last question to True
            questions_data[-1]["status"] = True
            questions_data[-1]["completed_at"] = datetime.now(IST)
            questions_data.append({
                "url": data.get('url', None),
                "question_number": 1 + len(questions_data),
                "status": False,
                "started_at": datetime.now(IST),
                "completed_at": None
            })
            # log all questions data in logging format
            print(f"--- QUESTIONS SUMMARY ---")
            for q in questions_data:
                print(f"Question {q['question_number']}: URL: {q['url']}, Status: {'Correct' if q['status'] else 'Incorrect'}, Started at: {q['started_at'].strftime('%Y-%m-%d %H:%M:%S')}, Completed at: {q['completed_at'].strftime('%Y-%m-%d %H:%M:%S') if q['completed_at'] else 'N/A'}")

        print("=== END EVALUATION ===\n")
        return {"status": "Evaluation completed.",
                "summary": questions_data,
                "email": "24f2000828@ds.study.iitm.ac.in"}
    
    return {"status": "Secret validated. Task is being processed in the background."}



















# def llm_process(data: dict) -> list[dict]:
#     """
#     Process task data through LLM API to generate code files.
#     Returns list of dicts with name and content for each file.
#     """
#     headers = {
#         "Authorization": f"Bearer {app.state.LLM_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     current_round = data.get('round', 1)

#     # System Instruction: Focused and strict
#     if current_round == 1:
#         system_instruction = (
#             "You are a strict, highly efficient code generation tool. "
#             "Generate ONLY the requested files. "
#             "DO NOT add any conversational text, explanations, or additional markdown outside the required file format. "
#             "Use the specified file format: <<FILENAME.ext>>[newline]<content>[newline]<<END_FILE>>"
#         )
#     else: # For Round 2 and beyond
#         # This instruction incorporates the allowance for new files
#         system_instruction = (
#             "You are a strict, highly efficient code refactoring and feature implementation tool. "
#             "Your task is to **UPDATE** the existing project files provided in the context to implement the new brief and pass all checks. "
#             "**PRIORITIZE UPDATING EXISTING FILES.** "
#             "**ONLY OUTPUT FILES THAT NEED MODIFICATION OR ARE NEWLY CREATED.** Do not output unchanged files. "
#             "DO NOT add any conversational text, explanations, or additional markdown outside the required file format. "
#             "Use the specified file format: <<FILENAME.ext>>[newline]<content>[newline]<<END_FILE>>"
#         )

#     if current_round == 1:
#         # User Prompt for Round 1 (Initial Generation)
#         prompt_goal = "Generate a complete, high-quality web app. Ensure all files work together seamlessly."
#     else:
#         # User Prompt for Round 2 (Update/Refactoring)
#         prompt_goal = (
#             f"UPDATE the existing web app (provided in the 'EXISTING CODE CONTEXT' below) to implement the new brief for Round 2. ONLY output the complete, updated content for files that require changes. You may generate **NEW FILES** if they are necessary to complete the task."
#         )

#     # User Prompt: Highly compressed
#     prompt = f"""
#     Task: {data.get('task')}
#     Brief: {data.get('brief')}
#     Round: {current_round}
#     Goal: {prompt_goal}
#     Checks: {data.get('checks')}
#     Files required: README.md (must include usage guide, cloning guide, inform License is MIT along with a message of being open to collaboration) , plus necessary HTML, CSS, JS.
#     """
    
#     # Attachments: Included as a dedicated, compressed block
#     attachments = data.get('attachments', [])
#     if attachments:
#         prompt += "\n--- ATTACHMENTS (File Name: URI) ---\n"
#         for attachment in attachments:
#             prompt += f"{attachment.get('name', 'N/A')}: {attachment.get('url', 'N/A')}\n"
#         prompt += "--- END ATTACHMENTS ---\n"

#     existing_code = data.get('existing_code_context')
#     if existing_code:
#         prompt += f"\n{existing_code}\n"
#         prompt += "Carefully review the existing code above. Your generated files in the output MUST be complete and correctly integrated with this existing code to implement the requested brief.\n"
    
#     # Output Instruction: Strict format definition (Critical for robustness)
#     prompt += """
    
#     Output all generated files using this format ONLY, starting immediately after this instruction:
    
#     <<FILENAME.ext>>
#     // File content goes here
#     <<END_FILE>>

#     Ensure no additional text, explanations, or markdown outside this format.
#     """

#     # API request payload
#     payload = {
#         "model": "openai/gpt-4.1-nano",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": system_instruction
#             },
#             {
#                 "role": "user", 
#                 "content": prompt
#             }
#         ],
#         "temperature": 0.2
#     }

#     try:
#         response = requests.post(
#             "https://aipipe.org/openrouter/v1/chat/completions",
#             headers=headers,
#             json=payload
#         )

#         if response.status_code != 200:
#             raise Exception(f"LLM API error: {response.status_code}, {response.text}")

#         # Parse LLM response and extract code files
#         content = response.json()["choices"][0]["message"]["content"]
#         files = extract_files_from_response(content)
#         print(f"LLM generated files successfully")
#         return files

#     except Exception as e:
#         print(f"Error in LLM processing: {str(e)}")
#         return []