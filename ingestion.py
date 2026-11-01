from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database import get_vectorstore, save_vectorstore
import os

def ingest_document(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
        
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
        
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents=splits)
    
    save_vectorstore(vectorstore)
    
    return True

def get_all_documents():
    vectorstore = get_vectorstore()

    if hasattr(vectorstore, "docstore") and hasattr(vectorstore.docstore, "_dict"):
        docs = list(vectorstore.docstore._dict.values())
        sources = set([doc.metadata.get("source", "Unknown") for doc in docs])
        return list(sources)
    return []