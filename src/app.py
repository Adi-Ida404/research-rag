from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import os
import shutil

from src.ingest import load_single_pdf, load_pdfs_from_folder
from src.embed_store import build_vectorstore, load_vectorstore
from src.rag_chain import build_qa_chain, ask_question

app = FastAPI(title="Research RAG")

# Path for storing uploaded PDFs and vectorstore
DATA_FOLDER = "data"
VECTORSTORE_PATH = "vectorstore"

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

class QuestionRequest(BaseModel):
    query: str
    
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
    
    return {"message": f"Uploaded {file.filename} and updated vectorstore."}

@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    """
    Ask a question to the RAG pipeline

    Args:
        request (QuestionRequest): The question to ask.
    """
    qa_chain = build_qa_chain(VECTORSTORE_PATH)
    answer = ask_question(request.query, qa_chain)
    return {"query": request.query, "answer": answer}