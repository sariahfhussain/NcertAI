 # NcertAI

 A PDF Question Answering API and educational tool using FastAPI, Google Gemini AI, and Groq Chat.

 ## Features
 - Upload PDF documents and split them into chunks
 - Ask questions about the content of uploaded PDFs
 - Generate multiple-choice quizzes for CBSE subjects and chapters
 - Evaluate quizzes and provide personalized feedback
 - General educational chat interface

 ## Prerequisites
 - Python 3.10 or higher
 - Git

 ## Installation
 1. Clone the repository:
    ```bash
    git clone https://github.com/mohdomer/NcertAI.git
    cd NcertAI
    ```
 2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate    # Windows
    ```
 3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

 ## Configuration
 Before running, update the API keys in `main.py`:
 ```python
 # Replace with your Google Generative AI API key
 API_KEY = "YOUR_GOOGLE_API_KEY"
 
 # Replace with your Groq API key
 groq_client = Groq(api_key="YOUR_GROQ_API_KEY")
 ```

 ## Running the API
## Running the Services
1. Run the PDF QA API (FastAPI)
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   - The API will be available at http://localhost:8000
   - Open http://localhost:8000/docs for interactive API docs.

2. Run the Web Application (Flask)
   ```bash
   cd NCERT-AI-main
   python app.py
   ```
   - The web app will be available at http://localhost:5000
   - It provides signup/login, chat, quiz, and PDF upload UI.

Ensure both services are running in separate terminals so they can communicate (the Flask app points to the FastAPI endpoints on port 8000).

 ## Client Usage
 A simple client example is provided in `client.py`. Update `BASE_URL` and `SESSION_ID` as needed, then run:
 ```bash
 python client.py
 ```

 ## API Endpoints
 - `GET /` - Health check
 - `POST /upload-pdf` - Upload and process a PDF file
 - `POST /ask-question` - Ask a question about an uploaded PDF
 - `POST /generate-quiz` - Generate a multiple-choice quiz for a chapter
 - `POST /evaluate-quiz` - Evaluate quiz answers and get feedback
 - `POST /educational-chat` - General educational chat
 - `DELETE /clear-session/{session_id}` - Clear a session
 - `GET /sessions` - List active sessions

 ## Contributing
 Contributions are welcome! Feel free to open issues or submit pull requests.

 ## License
 This project is available under the MIT License.