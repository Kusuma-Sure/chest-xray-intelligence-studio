"""Unit tests for Out-of-Distribution (OOD) and irrelevant image detection."""
import numpy as np
from PIL import Image
from backend.ood_detector import ood_detector


def test_reject_tiny_or_corrupt_image():
    tiny = Image.new("RGB", (32, 32), color=(128, 128, 128))
    res = ood_detector.analyze_image(tiny)
    assert not res["is_valid_cxr"]
    assert res["ood_score"] >= 0.55


def test_reject_blank_zero_contrast():
    blank = Image.new("RGB", (224, 224), color=(50, 50, 50))
    res = ood_detector.analyze_image(blank)
    assert not res["is_valid_cxr"]
    assert "contrast" in res["rejection_reason"].lower() or "dynamic range" in res["rejection_reason"].lower()


def test_reject_saturated_color_photo():
    # Simulated natural photograph (e.g. red sports car or animal) with high color saturation and contrast
    x = np.linspace(0, 255, 224, dtype=np.uint8)
    y = np.linspace(0, 255, 224, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)

    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(xx + 50, 0, 255)  # Red dominant
    arr[:, :, 1] = np.clip(yy // 4, 0, 255)  # Low green
    arr[:, :, 2] = 30                       # Low blue
    photo = Image.fromarray(arr)

    res = ood_detector.analyze_image(photo)
    assert not res["is_valid_cxr"]
    assert "color saturation" in res["rejection_reason"].lower()
    assert res["ood_score"] >= ood_detector.ood_threshold


def test_accept_synthetic_chest_radiograph():
    # Construct a synthetic grayscale image with bimodal thoracic profile and continuous texture
    # (dark lung fields, bright central mediastinum/heart/ribs)
    np.random.seed(42)
    base = np.random.normal(loc=120, scale=20, size=(224, 224)).clip(10, 240).astype(np.uint8)

    # Carve bilateral lung regions (darker, lower attenuation)
    base[30:190, 25:95] = (base[30:190, 25:95] * 0.4).astype(np.uint8)   # Right lung
    base[30:190, 129:199] = (base[30:190, 129:199] * 0.4).astype(np.uint8) # Left lung

    # Heart (lower left central opacity)
    base[100:180, 85:145] = np.clip(base[100:180, 85:145] * 1.4, 0, 250).astype(np.uint8)

    # Convert to 3-channel monochromatic RGB
    img = Image.fromarray(base).convert("RGB")
    res = ood_detector.analyze_image(img)
    assert res["is_valid_cxr"]
    assert res["ood_score"] < ood_detector.ood_threshold
