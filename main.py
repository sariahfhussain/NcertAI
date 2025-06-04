from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import os
import shutil
import json
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Question Answering API",
    description="Upload PDFs and ask questions about their content using Google's Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class QuestionResponse(BaseModel):
    answer: str
    session_id: str
    success: bool
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    session_id: str
    chunks_count: int
    success: bool
    error: Optional[str] = None
class QuizRequest(BaseModel):
    subject: str
    chapter: str

class QuizResponse(BaseModel):
    success: bool
    quiz: Optional[list] = None
    error: Optional[str] = None

class EvaluateQuizRequest(BaseModel):
    answers: list

class EvaluateQuizResponse(BaseModel):
    success: bool
    detailed_results: Optional[list] = None
    total_correct: Optional[int] = None
    total_questions: Optional[int] = None
    feedback: Optional[str] = None
    error: Optional[str] = None
 
# Educational Chat models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    success: bool
    error: Optional[str] = None

# Global configuration
API_KEY = 'AIzaSyAesqyliwOM5cGUKejbfLLTewG28ckIDgM'  # Replace with environment variable in production
# Configure Google Generative AI with API key to avoid ADC errors
genai.configure(api_key=API_KEY)
# Initialize Groq client for quiz generation
groq_client = Groq(api_key="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")

# Global storage for PDF processors (in production, use Redis or database)
pdf_processors = {}

class PDFQuestionAnswering:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)

        # Initialize Gemini model
        self.llm = GoogleGenerativeAI(
            model="models/gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )

    def load_pdf(self, pdf_path: str) -> tuple[bool, str, int]:
        """Load PDF and split into chunks"""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # Split pages into chunks
            chunks = self.text_splitter.split_documents(pages)
            chunks_count = len(chunks)
            
            logger.info(f"Successfully split PDF into {chunks_count} chunks")

            # Create vector store
            self.vectordb = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

            # Create QA chain
            prompt_template = """Use the following context to answer the question.
            If the answer cannot be found in the context, say "I don't know."
            Try to be as detailed as possible while staying true to the context If the user asks, explain the concept in a much simpler way, just as a teacher teaching a student.

            Context: {context}

            Question: {question}

            Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            return True, "PDF loaded successfully", chunks_count

        except Exception as e:
            error_msg = f"Error loading PDF: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, 0

    def ask_question(self, question: str) -> tuple[bool, str]:
        """Ask a question about the loaded PDF"""
        if not hasattr(self, 'qa_chain'):
            return False, "Please upload a PDF first!"

        try:
            result = self.qa_chain.invoke({"query": question})
            return True, result["result"]
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

# FastAPI Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PDF Question Answering API is running!", "status": "healthy"}

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(default="default")
):
    """
    Upload a PDF file and process it for question answering
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Copy uploaded file content to temporary file
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

        # Initialize PDF processor
        pdf_qa = PDFQuestionAnswering(API_KEY)
        
        # Load and process PDF
        success, message, chunks_count = pdf_qa.load_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if success:
            # Store processor for this session
            pdf_processors[session_id] = pdf_qa
            
            return UploadResponse(
                message=message,
                session_id=session_id,
                chunks_count=chunks_count,
                success=True
            )
        else:
            return UploadResponse(
                message="Failed to process PDF",
                session_id=session_id,
                chunks_count=0,
                success=False,
                error=message
            )

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded PDF
    """
    try:
        session_id = request.session_id or "default"
        
        # Check if PDF processor exists for this session
        if session_id not in pdf_processors:
            return QuestionResponse(
                answer="",
                session_id=session_id,
                success=False,
                error="No PDF uploaded for this session. Please upload a PDF first."
            )

        pdf_qa = pdf_processors[session_id]
        success, answer = pdf_qa.ask_question(request.question)

        return QuestionResponse(
            answer=answer,
            session_id=session_id,
            success=success,
            error=None if success else answer
        )

    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return QuestionResponse(
            answer="",
            session_id=request.session_id or "default",
            success=False,
            error=f"Internal server error: {str(e)}"
        )
@app.post("/generate-quiz", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    """
    Generate a multiple-choice quiz for the specified subject and chapter
    """
    try:
        subject = request.subject
        chapter = request.chapter
        prompt = (
            f"Generate a 10-question multiple choice quiz for CBSE Grade 9 {subject} Chapter {chapter}. "
            "Each question must have exactly 4 options, labeled A, B, C, and D. "
            "Respond ONLY with a JSON array (no additional text). "
            "The array should contain 10 objects, each with: "
            "'question' (string), 'options' (array of 4 strings in order A→D), and 'answer' (one of 'A','B','C','D')."
        )
        # Call Groq chat API to generate quiz
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        text = chat_completion.choices[0].message.content
        # Parse model output as JSON
        quiz = json.loads(text)
        return QuizResponse(success=True, quiz=quiz)
    except Exception as e:
        logger.error(f"Error in generate_quiz: {e}")
        return QuizResponse(success=False, error=str(e))

@app.post("/evaluate-quiz", response_model=EvaluateQuizResponse)
async def evaluate_quiz(request: EvaluateQuizRequest):
    """
    Evaluate a submitted quiz, return detailed results and personalized feedback.
    """
    try:
        answers = request.answers
        results = []
        correct_count = 0
        for item in answers:
            q_text = item.get("question")
            correct_ans = item.get("correct_answer")
            selected_ans = item.get("selected_answer")
            is_correct = (selected_ans == correct_ans)
            results.append({
                "question": q_text,
                "correct_answer": correct_ans,
                "selected_answer": selected_ans,
                "is_correct": is_correct
            })
            if is_correct:
                correct_count += 1
        total = len(answers)
        feedback_prompt = (
            "Here are the results of a quiz. Provide a personalized feedback message for the user, "
            "praising their correct answers and giving constructive advice for the questions they got wrong.\n\n"
        )
        for idx, r in enumerate(results, start=1):
            feedback_prompt += f"{idx}. Question: {r['question']}\n"
            feedback_prompt += f"   Your answer: {r['selected_answer']}\n"
            feedback_prompt += f"   Correct answer: {r['correct_answer']}\n"
            feedback_prompt += f"   {'Correct' if r['is_correct'] else 'Incorrect'}\n\n"
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful teacher giving personalized feedback to a student based on quiz results."},
                {"role": "user", "content": feedback_prompt}
            ],
            model="llama-3.3-70b-versatile"
        )
        feedback = chat_completion.choices[0].message.content
        return EvaluateQuizResponse(
            success=True,
            detailed_results=results,
            total_correct=correct_count,
            total_questions=total,
            feedback=feedback
        )
    except Exception as e:
        logger.error(f"Error in evaluate_quiz: {str(e)}")
        return EvaluateQuizResponse(success=False, error=str(e))

@app.post("/educational-chat", response_model=ChatResponse)
async def educational_chat(request: ChatRequest):
    """
    Provide educational chat responses using Groq
    """
    try:
        system_prompt = (
            "You are a helpful educational assistant. "
            "Answer user questions in an educational manner only, "
            "and do not deviate from educational content."
        )
        user_msg = request.message
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            model="llama-3.3-70b-versatile"
        )
        reply = chat_completion.choices[0].message.content
        return ChatResponse(reply=reply, success=True)
    except Exception as e:
        logger.error(f"Error in educational_chat: {str(e)}")
        return ChatResponse(reply="", success=False, error=str(e))

@app.delete("/clear-session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear a specific session and free up memory
    """
    try:
        if session_id in pdf_processors:
            del pdf_processors[session_id]
            return {"message": f"Session {session_id} cleared successfully", "success": True}
        else:
            return {"message": f"Session {session_id} not found", "success": False}
    except Exception as e:
        logger.error(f"Error in clear_session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    try:
        active_sessions = list(pdf_processors.keys())
        return {
            "active_sessions": active_sessions,
            "total_sessions": len(active_sessions)
        }
    except Exception as e:
        logger.error(f"Error in list_sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Development server runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )