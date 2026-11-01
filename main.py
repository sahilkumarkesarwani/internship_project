from fastapi import FastAPI, HTTPException, UploadFile, File
from models import QueryRequest, QueryResponse, IngestRequest, FeedbackRequest
from ingestion import ingest_document, get_all_documents
from graph import rag_application
from typing import Dict, List
import os
import uuid

app = FastAPI(title="Technical RAG Assistant")

chat_sessions: Dict[str, List[dict]] = {}

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)
    
    # Append UUID to prevent file collisions
    unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = f"data/{unique_filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    success = ingest_document(file_path)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process the document.")
        
    return {"message": f"Successfully uploaded and ingested {file.filename}"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    session_id = request.session_id
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        
    current_history = chat_sessions[session_id]
    
    initial_state = {
        "question": request.question,
        "chat_history": current_history,
        "documents": [],
        "answer": "",
        "web_search_needed": False,
        "is_grounded": True,
        "retry_count": 0  
    }
    
    result = rag_application.invoke(initial_state)
    
    chat_sessions[session_id].append({"role": "user", "content": request.question})
    chat_sessions[session_id].append({"role": "assistant", "content": result["answer"]})
    
    if len(chat_sessions[session_id]) > 6:
        chat_sessions[session_id] = chat_sessions[session_id][-6:]
    
    sources = []
    if result.get("documents"):
        for doc in result["documents"]:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)
                
    return {
        "question": result["question"], 
        "answer": result["answer"],
        "sources": sources
    }

@app.get("/documents")
async def list_documents_endpoint():
    return {"indexed_documents": get_all_documents()}

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    print(f"Feedback Q {request.request_id if hasattr(request, 'request_id') else ''}: {request.feedback} - {request.comment}")
    return {"message": "Feedback recorded"}