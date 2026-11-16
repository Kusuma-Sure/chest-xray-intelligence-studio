"""Unit tests for inference pipeline and model registry."""
from PIL import Image
from backend.config import FINDINGS, MODEL_INFO
from backend.model_registry import get_model_status, list_all_models
from backend.predictor import predictor


def test_model_registry_lists_all_four_models():
    models = list_all_models()
    assert len(models) == 4
    model_ids = {m["id"] for m in models}
    assert model_ids == {"vit", "densenet", "capsnet", "gat"}
    for m in models:
        assert "name" in m
        assert "status" in m
        assert "available" in m


def test_inference_handles_missing_checkpoint_gracefully():
    # If checkpoint is missing, should return informative status without mock predictions
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    res = predictor.predict_single_image(img, model_name="vit")

    # If checkpoint missing, success is False and user_message explains how to train
    if not get_model_status("vit")["available"]:
        assert res["success"] is False
        assert "model_unavailable" in res or "is_ood" in res
        assert "user_message" in res
        assert len(res["findings"]) == 0  # No mock predictions!


def test_batch_inference_handling():
    images = [Image.new("RGB", (224, 224), color=(i * 40, i * 40, i * 40)) for i in range(3)]
    batch_res = predictor.predict_batch(images, model_name="densenet")
    assert len(batch_res) == 3
    for idx, r in enumerate(batch_res):
        assert r["image_index"] == idx
