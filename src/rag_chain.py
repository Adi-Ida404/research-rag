from langchain_community.llms import HuggingFacePipeline 
from langchain.chains import RetrievalQA
from transformers import pipeline
from src.embed_store import load_vectorstore
import requests
import os
from dotenv import load_dotenv

# Load Hugging Face API key from .env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Choose your model
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_hf(prompt: str, max_new_tokens: int = 256):
    """
    Send prompt to Hugging Face Inference API and return the generated text

    Args:
        prompt (str): The input prompt to send to the model.
        max_new_tokens (int, optional): The maximum number of tokens to generate. Defaults to 256.

    Returns:
        str: The generated text from the model.
    """
    payload = {
        "inputs": prompt, 
        "parameters": {"max_new_tokens": max_new_tokens}
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {response.status_code}: {response.text}")

    result = response.json()
    return result[0]["generated_text"]

def build_prompt(query: str, vectorstore_path: str = "vectorstore", k: int= 4)-> str:
    """
    Builds a prompt using retrieved context from FAISS vectorstore.

    Args:
        query (str): The user query.
        vectorstore_path (str, optional): Path to the vectorstore. Defaults to "vectorstore".
        k (int, optional): Number of context documents to retrieve. Defaults to 4.

    Returns:
        str: The constructed prompt for the language model.
    """
    
    vectorstore = load_vectorstore(vectorstore_path)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are a research assistant. Use the following context to answer the question.
    Do not copy text directly, summarize and paraphrase in clear academic English.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    return prompt

def ask_question(query: str, vectorstore_path: str = "vectorstore") -> str:
    """
    Retrieve context, build prompt, send to Hugging Face API.
    """
    prompt = build_prompt(query, vectorstore_path)
    answer = query_hf(prompt)
    return answer
