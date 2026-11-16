"""Unit tests for Explainable AI (XAI) attribution methods and 4-panel grid."""
import numpy as np
import torch
from PIL import Image
from backend.architectures import build_model
from backend.explainability import (
    compute_gradcam_densenet,
    compute_gradcampp_capsnet,
    compute_gat_attention_map,
    generate_visual_explanation,
    normalize_map,
)
from backend.visualization import create_four_panel_grid


def test_normalize_map():
    arr = np.array([[10.0, 20.0], [30.0, 40.0]])
    norm = normalize_map(arr)
    assert norm.min() == 0.0
    assert norm.max() == 1.0


def test_gradcam_densenet_execution():
    model = build_model("densenet", pretrained=False)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    cam = compute_gradcam_densenet(model, x, target_class_idx=2)
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    assert 0.0 <= cam.min() and cam.max() <= 1.0


def test_gradcampp_capsnet_execution():
    model = build_model("capsnet", pretrained=False)
    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    cam = compute_gradcampp_capsnet(model, x, target_class_idx=1)
    assert isinstance(cam, np.ndarray)
    assert cam.ndim == 2
    assert 0.0 <= cam.min() and cam.max() <= 1.0


def test_gat_attention_map_execution():
    model = build_model("gat", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    attn = compute_gat_attention_map(model, x)
    assert isinstance(attn, np.ndarray)
    assert attn.shape == (7, 7)


def test_four_panel_grid_generation():
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    heatmap = np.random.rand(14, 14).astype(np.float32)
    rollout = np.random.rand(14, 14).astype(np.float32)

    panels = create_four_panel_grid(
        original_image=img,
        heatmap_raw=heatmap,
        rollout_raw=rollout,
        model_name="vit",
        finding_name="Cardiomegaly",
        probability=0.88,
    )
    assert "panel_1_original" in panels
    assert "panel_2_heatmap" in panels
    assert "panel_3_rollout" in panels
    assert "panel_4_fused" in panels
    assert "composite_grid" in panels
    assert panels["panel_1_original"].startswith("data:image/png;base64,")
