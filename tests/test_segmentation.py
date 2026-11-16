import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.api import app
from backend.segmentation import (
    segment_chest_anatomy,
    extract_anatomical_bounding_boxes,
    draw_bounding_boxes_on_image,
)
from backend.visualization import create_four_panel_grid


def test_segment_chest_anatomy():
    # Create synthetic CXR image (dark lung fields, brighter mediastinum)
    img_arr = np.ones((200, 200), dtype=np.uint8) * 160
    # Left and right lung ellipses (dark)
    img = Image.fromarray(img_arr, mode="L")
    anatomy = segment_chest_anatomy(img)

    assert "lung_mask" in anatomy
    assert "left_lung_mask" in anatomy
    assert "right_lung_mask" in anatomy
    assert "cardiac_mask" in anatomy
    assert "perihilar_mask" in anatomy
    assert "semantic_overlay" in anatomy

    assert anatomy["lung_mask"].shape == (200, 200)
    assert anatomy["cardiac_mask"].shape == (200, 200)
    assert anatomy["semantic_overlay"].shape == (200, 200, 3)
    assert anatomy["semantic_overlay"].dtype == np.uint8


def test_extract_anatomical_bounding_boxes():
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    anatomy = segment_chest_anatomy(img)

    # Synthetic heatmap with active peak in left lung
    heatmap = np.zeros((14, 14), dtype=np.float32)
    heatmap[8:12, 8:12] = 0.95

    boxes = extract_anatomical_bounding_boxes(
        heatmap=heatmap,
        anatomy=anatomy,
        finding_name="Lung Opacity",
        probability=0.85,
        orig_size=(200, 200),
    )

    assert isinstance(boxes, list)
    if boxes:
        b = boxes[0]
        assert "box" in b
        assert "label" in b
        assert "zone" in b
        assert len(b["box"]) == 4
        x1, y1, x2, y2 = b["box"]
        assert 0 <= x1 <= x2 <= 200
        assert 0 <= y1 <= y2 <= 200


def test_draw_bounding_boxes_on_image():
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    boxes = [{
        "box": [40, 50, 100, 120],
        "label": "Consolidation (82.1%)",
        "zone": "Right Mid Zone (RML)",
        "area_pct": 5.2,
        "peak_norm": 0.92,
    }]
    annotated = draw_bounding_boxes_on_image(img, boxes)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (200, 200)
    assert annotated.mode == "RGB"


def test_four_panel_grid_with_anatomy():
    img = Image.new("RGB", (200, 200), color=(80, 80, 80))
    heatmap = np.random.uniform(0.0, 1.0, (14, 14)).astype(np.float32)
    panels = create_four_panel_grid(
        original_image=img,
        heatmap_raw=heatmap,
        rollout_raw=None,
        model_name="vit",
        finding_name="Pneumonia",
        probability=0.74,
    )

    assert "panel_1_original" in panels
    assert "panel_1_annotated" in panels
    assert "panel_2_anatomy" in panels
    assert "panel_2_binary_lung" in panels
    assert "panel_3_rollout" in panels
    assert "panel_4_fused" in panels
    assert "composite_grid" in panels
    assert "bounding_boxes" in panels


def test_api_models_analysis_endpoint():
    client = TestClient(app)
    resp = client.get("/api/models/analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis" in data
    analysis = data["analysis"]
    for arch in ["vit", "densenet", "capsnet", "gat"]:
        assert arch in analysis
        info = analysis[arch]
        assert "recommended_threshold" in info
        assert "optimism_bias" in info
        assert "hidden_dependencies" in info
        assert "unclear_scope" in info
        assert isinstance(info["recommended_threshold"], float)
        assert 0.30 <= info["recommended_threshold"] <= 0.50
