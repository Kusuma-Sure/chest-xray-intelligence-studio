"""Visualization and Image Fusion Module for Medical AI Attribution.

Generates:
1. High-contrast colormap heatmaps (JET / Turbo / Viridis)
2. Fused alpha overlays on original radiographs
3. Complete 4-Panel Diagnostic Grid:
   - Panel 1: Original Chest Radiograph
   - Panel 2: Model-Specific Heatmap (Grad-CAM, Grad-CAM++, ViT Grad×Act, GAT Region)
   - Panel 3: ViT Attention Rollout (for ViT) or Secondary Structural Attribution
   - Panel 4: Fused Diagnostic Visual Overlay
4. Base64 Data-URL encoding for real-time frontend rendering
"""
from __future__ import annotations
import base64
import io
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 Data-URL string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{encoded}"


def apply_colormap_to_heatmap(heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resizes a 2D [0, 1] normalized attribution map and applies JET colormap (RGB uint8)."""
    w, h = target_size
    resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)

    # Convert to 0-255 uint8 and apply JET colormap
    uint8_map = np.uint8(255 * resized)
    color_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb


def create_fused_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    beta: float = 0.55,
) -> Image.Image:
    """Blends the original radiograph with the colorized attribution heatmap."""
    w, h = original_image.size
    orig_rgb = np.array(original_image.convert("RGB"), dtype=np.uint8)
    color_map = apply_colormap_to_heatmap(heatmap, (w, h))

    blended = cv2.addWeighted(orig_rgb, beta, color_map, alpha, 0)
    return Image.fromarray(blended)


def create_four_panel_grid(
    original_image: Image.Image,
    heatmap_raw: np.ndarray,
    rollout_raw: np.ndarray | None,
    model_name: str,
    finding_name: str,
    probability: float,
) -> dict:
    """Constructs the comprehensive 4-panel diagnostic visual explanation grid.

    Returns:
        dict containing base64 data URLs for:
            - panel_1_original
            - panel_2_heatmap
            - panel_3_rollout (or secondary view)
            - panel_4_fused
            - composite_grid (the unified 4-panel figure)
    """
    w, h = original_image.size
    orig_rgb = original_image.convert("RGB")

    # Panel 1: Original Radiograph
    panel_1_url = image_to_data_url(orig_rgb)

    # Panel 2: Model-specific Heatmap
    color_heatmap = apply_colormap_to_heatmap(heatmap_raw, (w, h))
    panel_2_img = Image.fromarray(color_heatmap)
    panel_2_url = image_to_data_url(panel_2_img)

    # Panel 3: ViT Attention Rollout (or GAT/CapsNet structural secondary view)
    if rollout_raw is not None:
        color_rollout = apply_colormap_to_heatmap(rollout_raw, (w, h))
        panel_3_img = Image.fromarray(color_rollout)
        panel_3_title = "ViT Attention Rollout (12 Layers)"
        panel_3_available = True
    else:
        # Generate edge-enhanced anatomical structure view for CNN / GAT / CapsNet
        orig_gray = np.array(original_image.convert("L"))
        edges = cv2.Canny(orig_gray, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        panel_3_img = Image.fromarray(edges_color)
        panel_3_title = "Anatomical Edge Boundaries (Structural Reference)"
        panel_3_available = False
    panel_3_url = image_to_data_url(panel_3_img)

    # Panel 4: Fused Diagnostic Visual Overlay
    panel_4_img = create_fused_overlay(original_image, heatmap_raw, alpha=0.42, beta=0.58)
    panel_4_url = image_to_data_url(panel_4_img)

    # Generate unified composite figure (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d1117")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    # Panel 1
    axes[0].imshow(np.array(orig_rgb))
    axes[0].set_title("Panel 1: Original X-ray", color="#e6edf3", fontsize=12, pad=8, fontweight="bold")

    # Panel 2
    axes[1].imshow(color_heatmap)
    method_name = {
        "vit": "ViT Grad × Activation",
        "densenet": "DenseNet121 Grad-CAM",
        "capsnet": "CapsNet Grad-CAM++",
        "gat": "GAT Regional Attention",
    }.get(model_name.lower(), "Model Attribution")
    axes[1].set_title(f"Panel 2: {method_name}", color="#e6edf3", fontsize=12, pad=8, fontweight="bold")

    # Panel 3
    axes[2].imshow(np.array(panel_3_img))
    axes[2].set_title(f"Panel 3: {panel_3_title}", color="#e6edf3", fontsize=12, pad=8, fontweight="bold")

    # Panel 4
    axes[3].imshow(np.array(panel_4_img))
    axes[3].set_title(
        f"Panel 4: Fused Overlay\n{finding_name} ({probability:.1%})",
        color="#00d2ff",
        fontsize=12,
        pad=8,
        fontweight="bold",
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    composite_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    return {
        "panel_1_original": panel_1_url,
        "panel_2_heatmap": panel_2_url,
        "panel_3_rollout": panel_3_url,
        "panel_3_title": panel_3_title,
        "panel_3_available": panel_3_available,
        "panel_4_fused": panel_4_url,
        "composite_grid": composite_url,
        "finding": finding_name,
        "probability": float(probability),
    }
