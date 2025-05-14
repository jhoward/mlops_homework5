from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_similar_responses_endpoint():
    response = client.post("/api/similar_responses", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert isinstance(data["answers"], list)
    if len(data["answers"]) > 0:
        answer = data["answers"][0]
        assert "prompt" in answer
        assert "excerpt" in answer
        assert "answer" in answer
        assert "similarity_score" in answer
        assert isinstance(answer["similarity_score"], float)

def test_similar_responses_empty_question():
    response = client.post("/api/similar_responses", json={"question": ""})
    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert isinstance(data["answers"], list)

def test_similar_responses_invalid_json():
    response = client.post("/api/similar_responses", json={"invalid": "data"})
    assert response.status_code == 422  # Validation error

