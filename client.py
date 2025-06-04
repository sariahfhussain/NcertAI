import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
SESSION_ID = "my_session_1"

def upload_pdf(file_path: str, session_id: str = SESSION_ID):
    """Upload a PDF file to the API"""
    url = f"{BASE_URL}/upload-pdf"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'session_id': session_id}
        
        response = requests.post(url, files=files, data=data)
        
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print(f"✅ PDF uploaded successfully!")
            print(f"Session ID: {result['session_id']}")
            print(f"Chunks created: {result['chunks_count']}")
            return True
        else:
            print(f"❌ Upload failed: {result['error']}")
            return False
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        return False

def ask_question(question: str, session_id: str = SESSION_ID):
    """Ask a question about the uploaded PDF"""
    url = f"{BASE_URL}/ask-question"
    
    payload = {
        "question": question,
        "session_id": session_id
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print(f"\n🤖 Answer: {result['answer']}")
            return result['answer']
        else:
            print(f"❌ Question failed: {result['error']}")
            return None
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        return None

def list_sessions():
    """List all active sessions"""
    url = f"{BASE_URL}/sessions"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Active sessions: {result['active_sessions']}")
        print(f"Total sessions: {result['total_sessions']}")
        return result
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        return None

def clear_session(session_id: str = SESSION_ID):
    """Clear a specific session"""
    url = f"{BASE_URL}/clear-session/{session_id}"
    response = requests.delete(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Session cleared: {result['message']}")
        return result['success']
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        return False

def main():
    """Example usage of the PDF QA API"""
    print("🚀 PDF Question Answering API Client")
    print("=" * 40)
    
    # Example 1: Upload a PDF
    pdf_path = "your_document.pdf"  # Replace with your PDF path
    print(f"\n1. Uploading PDF: {pdf_path}")
    
    if upload_pdf(pdf_path):
        
        # Example 2: Ask questions
        questions = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the conclusions mentioned?"
        ]
        
        print("\n2. Asking questions:")
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")
            ask_question(question)
        
        # Example 3: List sessions
        print("\n3. Listing active sessions:")
        list_sessions()
        
        # Example 4: Clear session
        print(f"\n4. Clearing session: {SESSION_ID}")
        clear_session()

if __name__ == "__main__":
    main()