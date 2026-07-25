from fastapi.testclient import TestClient
from src.api.app import app
import json
from pathlib import Path

client = TestClient(app)
ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = ROOT / "sample.json"

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Message": "Welcome to Handwritten Digit Classifier API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict():
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        sample = json.load(f)
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data



