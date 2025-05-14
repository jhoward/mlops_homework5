from pydantic import BaseModel
from typing import List, Dict

class RAGRequest(BaseModel):
    question: str

class Response(BaseModel):
    prompt: str
    excerpt: str
    answer: str
    similarity_score: float

class RAGResponse(BaseModel):
    answers: List[Response]
