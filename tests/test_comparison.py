from __future__ import annotations
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.api import app
from backend.predictor import predictor


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def synthetic_cxr():
    """Generates a valid grayscale synthetic chest radiograph for testing."""
    arr = np.zeros((224, 224), dtype=np.uint8)
    # Lungs (darker bilateral regions)
    arr[40:180, 30:95] = 40
    arr[40:180, 129:194] = 40
    # Mediastinum / spine (brighter center)
    arr[:, 95:129] = 170
    # Soft tissue borders
    arr[0:40, :] = 130
    arr[180:224, :] = 130
    # Add subtle Gaussian texture
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 5, (224, 224)).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def test_compare_all_models_executes_all_four(synthetic_cxr):
    """Verify that compare_all_models runs all 4 architectures independently."""
    result = predictor.compare_all_models(image=synthetic_cxr)
    
    assert result["success"] is True
    assert result["is_ood"] is False
    assert len(result["models"]) == 4
    
    expected_ids = {"vit", "densenet", "capsnet", "gat"}
    returned_ids = {m["model_id"] for m in result["models"]}
    assert returned_ids == expected_ids
    
    for m in result["models"]:
        assert "model_name" in m
        assert "primary_conclusion" in m
        assert "primary_confidence" in m
        assert "processing_time_ms" in m
        assert "alternatives" in m
        assert isinstance(m["should_abstain"], bool)

    # Check consensus structure
    consensus = result["consensus"]
    assert "level" in consensus
    assert "title" in consensus
    assert "badge" in consensus
    assert "agreement_count" in consensus
    assert isinstance(consensus["agreement_count"], int)
    assert 0 <= consensus["agreement_count"] <= 4


def test_compare_all_models_ood_rejection():
    """Verify that compare_all_models rejects out-of-distribution non-CXR images."""
    tiny_img = Image.new("RGB", (16, 16), color="red")
    result = predictor.compare_all_models(image=tiny_img)
    
    assert result["success"] is False
    assert result["is_ood"] is True
    assert result["consensus"]["level"] == "rejected"
    assert "Input Out-of-Distribution" in result["consensus"]["badge"]


def test_api_compare_endpoint(test_client, synthetic_cxr):
    """Test the POST /api/compare endpoint via TestClient."""
    buf = io.BytesIO()
    synthetic_cxr.save(buf, format="PNG")
    buf.seek(0)
    
    response = test_client.post(
        "/api/compare",
        files={"image": ("synthetic_cxr.png", buf, "image/png")},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "models" in data
    assert len(data["models"]) == 4
    assert "consensus" in data
    assert "discrepancies" in data


def test_api_compare_corrupt_image(test_client):
    """Test the POST /api/compare endpoint with corrupted image bytes."""
    corrupted_bytes = io.BytesIO(b"Not a valid image data payload")
    response = test_client.post(
        "/api/compare",
        files={"image": ("corrupted.png", corrupted_bytes, "image/png")},
    )
    assert response.status_code == 400


def test_api_compare_unsupported_format(test_client):
    """Test the POST /api/compare endpoint with unsupported media type."""
    text_bytes = io.BytesIO(b"Some text document")
    response = test_client.post(
        "/api/compare",
        files={"image": ("document.txt", text_bytes, "text/plain")},
    )
    assert response.status_code == 415
