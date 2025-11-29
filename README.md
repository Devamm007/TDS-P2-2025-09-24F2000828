# 🤖 LLM-Powered Web Analysis and Quiz Solver

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Playwright](https://img.shields.io/badge/Browser-Playwright-2EAD33.svg)](https://playwright.dev/)

An intelligent automated system that scrapes, analyzes, and solves web-based quizzes using Large Language Models (LLMs). This project leverages FastAPI, Playwright for headless browsing, and multimodal AI capabilities to process various content types including text, images, audio, PDFs, and more.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
  - [Using pip](#using-pip)
  - [Using uv](#using-uv-recommended)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

### 🔍 Multi-Format Content Extraction
- **HTML Scraping**: Full page content extraction with Playwright
- **Images**: AI-powered image description using GPT-4o-mini
- **Audio**: Speech-to-text transcription for MP3, WAV, FLAC, AAC, OGG, M4A, ALAC, WMA
- **PDFs**: Text and table extraction using pdfplumber
- **Documents**: Support for DOCX, PPTX, XLSX files
- **Archives**: Automated extraction from ZIP, TAR, TAR.GZ, TGZ files
- **Videos**: Detection and cataloging of video files
- **Linked Pages**: Recursive content extraction from nested URLs

### 🧠 Intelligent Processing
- **LLM Integration**: Powered by GPT-4o-mini for question understanding and answer generation
- **Multimodal Analysis**: Combines text, visual, and audio data for comprehensive understanding
- **Iterative Solving**: Automatic retry mechanism with context-aware error handling
- **Screenshot Analysis**: Visual page understanding for complex layouts

### 🚀 Robust Architecture
- **Async Processing**: Non-blocking background task execution
- **Headless Browser**: Playwright-based automation for JavaScript-rendered content
- **RESTful API**: Clean FastAPI endpoints with CORS support
- **Error Handling**: Comprehensive error catching and logging
- **Time Management**: Built-in timeout and iteration limits

## 🏗️ Architecture

```
┌─────────────┐
│  Client     │
│  Request    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  FastAPI Server     │
│  /handle_task       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Worker Process     │
│  (Background Task)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐      ┌──────────────────┐
│  Playwright Scraper │◄────►│  Content Parser  │
│  (Headless Browser) │      │  (Multi-format)  │
└──────┬──────────────┘      └────────┬─────────┘
       │                              │
       ▼                              ▼
┌─────────────────────────────────────────┐
│           LLM Analyzer                  │
│  (GPT-4o-mini, GPT-4o-audio-preview)   │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Answer Submission  │
│  & Result Handling  │
└─────────────────────┘
```

## 📦 Prerequisites

- **Python**: 3.11 or higher
- **System Dependencies**: For Playwright browser automation
- **API Keys**: 
  - LLM API key (OpenRouter/OpenAI compatible)
  - Optional: Gemini API key

## 🛠️ Installation

### Using pip

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd P2
```

2. **Create a virtual environment**
```bash
python -m venv .venv

# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Playwright browsers**
```bash
playwright install chromium
playwright install-deps chromium
```

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a blazing fast Python package installer and resolver.

1. **Install uv** (if not already installed)
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Clone the repository**
```bash
git clone <your-repo-url>
cd P2
```

3. **Create virtual environment and install dependencies**
```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

uv pip install -r requirements.txt
```

4. **Install Playwright browsers**
```bash
playwright install chromium
playwright install-deps chromium
```

## ⚙️ Configuration

1. **Create a `.env` file** in the project root:

```env
# Required
LLM_API_KEY=your_llm_api_key_here
SECRET=your_secret_key_here
EMAIL=your_email@example.com
LLM_BASE_URL=https://aipipe.org/openrouter/v1
LLM_MODEL=gpt-4o-mini

# Optional
GEMINI_API_KEY=your_gemini_api_key_here
```

2. **Environment Variables Explained**:
   - `LLM_API_KEY`: Your OpenRouter or OpenAI-compatible API key
   - `SECRET`: Secret key for API authentication
   - `EMAIL`: Your registered email for validation
   - `LLM_BASE_URL`: Base URL for LLM API (default: OpenRouter)
   - `LLM_MODEL`: Model to use for analysis (default: gpt-4o-mini)

## 🚀 Usage

### Local Development

1. **Start the server**
```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Access the web interface**
```
http://localhost:8000
```

3. **API endpoint** for task submission:
```bash
curl -X POST "http://localhost:8000/handle_task" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your_email@example.com",
    "secret": "your_secret_key",
    "url": "https://example.com/quiz"
  }'
```

### Response Format

```json
{
  "status": "Secret validated. Task is being processed in the background."
}
```

The task runs asynchronously. Check server logs for detailed progress.

## 🌐 Deployment

### Docker (Hugging Face Spaces)

1. **Build the Docker image**
```bash
docker build -t llm-quiz-solver .
```

2. **Run the container**
```bash
docker run -p 7860:7860 --env-file .env llm-quiz-solver
```

3. **Deploy to Hugging Face Spaces**
   - Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - Select "Docker" as SDK
   - Push your code including `Dockerfile`
   - Set environment variables in Space settings (don't commit `.env`)

### Production Considerations

- Use environment variables instead of `.env` file in production
- Set up proper logging and monitoring
- Configure rate limiting for API endpoints
- Use a reverse proxy (nginx) for SSL termination
- Consider containerization with Docker Compose for multi-service setups

## 📚 API Documentation

### Endpoints

#### `GET /`
Returns the web interface (index.html)

#### `POST /handle_task`

**Request Body:**
```json
{
  "email": "string",
  "secret": "string",
  "url": "string (https only)"
}
```

**Response:**
```json
{
  "status": "string"
}
```

**Error Responses:**
- `400`: Invalid JSON structure, invalid email, or invalid URL
- `403`: Invalid secret

### Interactive API Docs

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📁 Project Structure

```
P2/
├── main.py                 # FastAPI application and core logic
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration for deployment
├── .dockerignore          # Docker build exclusions
├── .env                   # Environment variables (not in git)
├── .gitignore             # Git exclusions
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface
└── __pycache__/           # Python cache (auto-generated)
```

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🌟 Ways to Contribute

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Share your ideas for new functionality
3. **Improve Documentation**: Help make our docs clearer and more comprehensive
4. **Submit Pull Requests**: Fix bugs or implement new features
5. **Share Use Cases**: Tell us how you're using this project
6. **Optimize Performance**: Help us make the code faster and more efficient

### 🔧 Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable
4. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### 💡 Areas for Contribution

- **New Content Parsers**: Add support for more file formats
- **LLM Integrations**: Add support for other LLM providers
- **Performance**: Optimize scraping and processing speed
- **Security**: Enhance authentication and rate limiting
- **Testing**: Add unit and integration tests
- **UI/UX**: Improve the web interface
- **Documentation**: Add tutorials, examples, and guides

### 📝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other contributors

### 🎓 Learning Opportunity

**Important Note**: *If you're here to learn, don't just copy-paste the code. Understanding the architecture, debugging issues, and implementing features yourself will give you valuable skills that will set you apart in your career. Take time to experiment, break things, and fix them. That's where real learning happens!*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI**: For the amazing web framework
- **Playwright**: For robust browser automation
- **OpenAI**: For powerful LLM capabilities
- **OpenRouter**: For LLM API access
- **Hugging Face**: For deployment infrastructure
- **Contributors**: Everyone who helps improve this project

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/yourrepo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/yourrepo/discussions)
- **Email**: 24f2000828@ds.study.iitm.ac.in

---

**⭐ If this project helped you, consider giving it a star!**

**🤝 We're excited to collaborate with you and build something amazing together!**

---

*Built with ❤️ by Devam*
