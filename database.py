import os
import faiss
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
PATH = "database"

def get_vectorstore():
    index_file = os.path.join(PATH, "index.faiss")
    
    if os.path.exists(index_file):
        return FAISS.load_local(
            PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("init")))
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

def save_vectorstore(vectorstore):
    vectorstore.save_local(PATH)

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})