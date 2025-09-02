import os
import boto3
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# AWS client
s3 = boto3.client("s3")


def build_vectorstore(documents, persist_path: str="vectorstore"):
    """
    Splits documents, embeds them and creates a vector store.
    Saves the vector store path so that it can be used later.

    Args:
        documents (list): A list of documents to embed and store.
        persist_path (str, optional): The path to persist the vector store. Defaults to "vectorstore".
        
    """
    # 1. Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_documents(documents)
    
    # 2. Embed Model (small + fast for testing)
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 3. Build FAISS Index
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. Save locally
    if persist_path:
        vector_store.save_local(persist_path)

    return vector_store

def upload_vector_to_s3(local_path: str, bucket: str, key: str):
    """
    Uploads a saved FAISS vectorsstore to s3

    Args:
        local_path (str): The local path to the FAISS vector store.
        bucket (str): The S3 bucket to upload the vector store to.
        key (str): The S3 key (path) to upload the vector store to.
    """
    for root, _, files in os.walk(local_path):
       for file in files:
           s3.upload_file(
               os.path.join(root,file),
               bucket,
               f"{key}/{file}"
           )

def download_vectorstore_from_s3(bucket: str, key: str, local_path: str="vectorstore"):
    """
    Downloads a FAISS vector store from S3.

    Args:
        bucket (str): The S3 bucket to download the vector store from.
        key (str): The S3 key (path) to download the vector store from.
        local_path (str, optional): The local path to save the downloaded vector store. Defaults to "vectorstore".
    """
    os.makedirs(local_path, exist_ok=True)
    objects = s3.list_objects_v2(Bucket=bucket, prefix=key)
    
    if "Contents" not in objects:
        raise FileNotFoundError(f"No objects found in bucket {bucket} with prefix {key}")
    
    for obj in objects["Contents"]:
        filename = obj["Key"].split("/")[-1]
        s3.download_file(bucket, obj["Key"], os.path.join(local_path, filename))
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(local_path, embeddings, allow_dangerous_deserialization=True)
