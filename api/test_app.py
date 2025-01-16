import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API en ligne"}

def test_prediction():
    with open("data/X_test.npy", "rb") as f:
        response = client.post("/predict", files={"file": f})
    assert response.status_code == 200
    assert "prediction" in response.json()
