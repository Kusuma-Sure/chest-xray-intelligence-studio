"""Unit tests for FastAPI endpoints and server routing."""
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)


def test_home_page_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "Chest X-ray Intelligence" in res.text


def test_api_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["models_total"] == 4


def test_api_models_endpoint():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) == 4
    model_ids = [m["id"] for m in data["models"]]
    assert "vit" in model_ids
    assert "densenet" in model_ids
    assert "capsnet" in model_ids
    assert "gat" in model_ids


def test_api_benchmarks_endpoint():
    res = client.get("/api/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "comparison_chart" in data
    assert "provenance" in data


def test_api_graph_evidence_endpoint():
    res = client.get("/api/graph-evidence?finding=Cardiomegaly")
    assert res.status_code == 200
    data = res.json()
    assert "grounded_explanation" in data
    assert "evidence" in data
