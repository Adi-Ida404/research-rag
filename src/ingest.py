import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

def load_single_pdf(file_path: str):
    """
    Loads a single path into the LangChain format

    Args:
        file_path (str): _description_
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document

def load_pdfs_from_folder(file_path: str):
    """
    Loads all PDF files inside a given folder
    
    Args:
        file_path (str): _description_
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The folder {file_path} does not exist.")
    
    loader = DirectoryLoader(file_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents