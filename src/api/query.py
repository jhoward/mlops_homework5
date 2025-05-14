from fastapi import APIRouter
from src.retriever.retriever import Retriever
from src.models.query import RAGRequest, RAGResponse

router = APIRouter()
retriever = Retriever()  # Create an instance of the Retriever

@router.post("/similar_responses", response_model=RAGResponse)
def get_similar_responses(request: RAGRequest):
    results = retriever.get_similar_responses(request.question)
    return RAGResponse(answers=results)
