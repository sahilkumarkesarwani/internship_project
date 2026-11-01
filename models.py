from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default" 

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

class IngestRequest(BaseModel):
    file_path: str

class FeedbackRequest(BaseModel):
    question_id: str
    feedback: str
    comment: Optional[str] = None