from langchain_community.llms import HuggingFacePipeline 
from langchain.chains import RetrievalQA
from transformers import pipeline
from src.embed_store import load_vectorstore
import requests
import os

HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HF_API_KEY = os.getenv("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def hf_query(prompt: str, max_new_tokens: int = 256):
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()
    return result[0]["generated_text"]

def load_model(model_id="google/flan-t5-base", max_new_tokens=256):
    pipe = pipeline(
        "text2text-generation",
        model=model_id,
        tokenizer=model_id,
        max_new_tokens=max_new_tokens,
        device_map="auto"   # uses CPU if no GPU
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_qa_chain(vectorstore_path: str = "vectorstore"):
    """
    Builds a RetrievalQA chain using Falcon + FAISS vectorstore.

    Args:
        vectorstore_path (str): Path to the vectorstore.
    """
    # Load vector store
    vectorstore = load_vectorstore(vectorstore_path)

    # Load model
    llm = load_model()
    
    # Create RetrievalQA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff"
    )

    
    return qa_chain

def ask_question(query: str, qa_chain):
    """
    Ask a question to the RAG pipeline and return the response.

    Args:
        qa_chain (_type_): The QA chain to use for asking questions.
        question (_type_): The question to ask.
    """
    
    response = qa_chain.run(query)
    return response