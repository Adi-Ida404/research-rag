from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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

def load_vectorstore(persist_path: str="vectorstore"):
    """
    Reloads a previously saved FAISS vector store.

    Args:
        persist_path (str, optional): The path to the persisted vector store. Defaults to "vectorstore".
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization = True)