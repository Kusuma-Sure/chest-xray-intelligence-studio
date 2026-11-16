from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.graph_rag import search_medical_evidence


@pytest.fixture
def test_client():
    return TestClient(app)


def test_search_medical_evidence_known_finding():
    """Verify evidence retrieval for closed-set CheXpert findings."""
    evidence = search_medical_evidence("Cardiomegaly")
    assert evidence["found"] is True
    assert evidence["finding"] == "Cardiomegaly"
    assert "Cardiomegaly" in evidence["title"] or "Cardiac" in evidence["title"]
    assert "authors" in evidence
    assert "source_url" in evidence
    assert "pubmed_search_url" in evidence
    assert "Cardiac" in evidence["anatomical_region"] or "Silhouette" in evidence["anatomical_region"]
    assert "informational/literature context" in evidence["disclaimer"]


def test_search_medical_evidence_expanded_finding():
    """Verify evidence retrieval for expanded open-set thoracic findings."""
    evidence = search_medical_evidence("Pneumoperitoneum")
    assert evidence["found"] is True
    assert evidence["finding"] == "Pneumoperitoneum"
    assert "Pneumoperitoneum" in evidence["title"]
    assert "Subdiaphragmatic" in evidence["anatomical_region"]
    assert "crescentic" in evidence["visual_concept"].lower() or "free" in evidence["visual_concept"].lower()
    assert "source_url" in evidence


def test_search_medical_evidence_case_insensitivity():
    """Verify search handles lowercase and partial names gracefully."""
    ev1 = search_medical_evidence("emphysema")
    assert ev1["found"] is True
    assert ev1["finding"] == "Emphysema"
    assert "Lung Parenchyma" in ev1["anatomical_region"]

    ev2 = search_medical_evidence("tuberculosis")
    assert ev2["found"] is True
    assert ev2["finding"] == "Tuberculosis"


def test_api_medical_evidence_endpoint(test_client):
    """Test GET /api/medical-evidence endpoint."""
    response = test_client.get("/api/medical-evidence?finding=Pneumothorax")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["finding"] == "Pneumothorax"
    assert "source_url" in data
    assert "pubmed_search_url" in data
    assert "disclaimer" in data


def test_api_ontology_endpoint(test_client):
    """Test GET /api/ontology endpoint returns the full 28+ thoracic condition catalog."""
    response = test_client.get("/api/ontology")
    assert response.status_code == 200
    data = response.json()
    assert data["total_conditions"] >= 28
    assert len(data["chexpert_findings"]) == 13
    assert len(data["expanded_findings"]) >= 15
    assert "anatomy_metadata" in data
    assert "disclaimer" in data
