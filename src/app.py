from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv

from src.ingest import load_pdfs_from_folder
from src.embed_store import build_vectorstore, upload_vectorstore_to_s3, download_vectorstore_from_s3
from src.rag_chain import ask_question

# Load environment variables
load_dotenv()

app = FastAPI(title="Research RAG")

# Path for storing uploaded PDFs and vectorstore
DATA_FOLDER = "data"
VECTORSTORE_PATH = "vectorstore"

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# S3 Config
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = "vectorstore"

class QuestionRequest(BaseModel):
    query: str
    

@app.get("/")
async def root():
    """
    Root endpoint for the API.

    Returns:
        _type_: _description_
    """
    return {
        "message": "Welcome to the Research RAG API",
        "docs": "/docs",
        "health": "/health"
    }
    
@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns 200 if app is alive.
    """
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Upload a PDF file and rebuild vectorstore.

    Args:
        file (UploadFile): The PDF file to upload.
    """
    # Save Uploaded file
    file_path = os.path.join(DATA_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Load all PDFs and rebuild vectorstore
    docs = load_pdfs_from_folder(DATA_FOLDER)
    build_vectorstore(docs, persist_path=VECTORSTORE_PATH)
    
    #Upload vectorstore to S3
    upload_vectorstore_to_s3(VECTORSTORE_PATH, S3_BUCKET, S3_KEY)
    
    return {"message": f"Uploaded {file.filename} and updated vectorstore in S3."}

@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    """
    Ask a question to the RAG pipeline (vectorstore pulled from S3)

    Args:
        request (QuestionRequest): The question to ask.
    """
    # Always get the latest vectorstore from S3
    download_vectorstore_from_s3(S3_BUCKET, S3_KEY, VECTORSTORE_PATH)
    
    # Run retrieval + Hugging Face API inference
    answer = ask_question(request.query, VECTORSTORE_PATH)
    
    return {"query": request.query, "answer": answer}