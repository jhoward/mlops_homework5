import os
import pytest
from src.retriever.retriever import Retriever

@pytest.fixture
def retriever():
    # Create a test instance of Retriever
    retriever = Retriever()
    return retriever

def test_get_similar_responses(retriever):
    results = retriever.get_similar_responses("What is the capital of France?")
    assert isinstance(results, list)
    if len(results) > 0:
        result = results[0]
        assert "prompt" in result
        assert "excerpt" in result
        assert "answer" in result
        assert "similarity_score" in result
        assert isinstance(result["similarity_score"], float)

def test_get_similar_responses_empty_query(retriever):
    results = retriever.get_similar_responses("")
    assert isinstance(results, list)

def test_get_similar_responses_k_parameter(retriever):
    k = 3
    results = retriever.get_similar_responses("What is the capital of France?", k=k)
    assert isinstance(results, list)
    assert len(results) <= k
    
    


