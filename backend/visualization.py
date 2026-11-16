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


from .segmentation import (
    draw_bounding_boxes_on_image,
    extract_anatomical_bounding_boxes,
    segment_chest_anatomy,
)


def create_fused_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    target_mask: np.ndarray | None = None,
    alpha: float = 0.45,
    beta: float = 0.55,
) -> Image.Image:
    """Blends the original radiograph with the colorized attribution heatmap.

    Optionally applies an anatomical soft-mask to suppress non-thoracic background artifacts.
    """
    w, h = original_image.size
    orig_rgb = np.array(original_image.convert("RGB"), dtype=np.uint8)

    # Resize raw heatmap to match image dimensions (w, h)
    if heatmap.shape != (h, w):
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
    else:
        heatmap_resized = heatmap

    if target_mask is not None:
        if target_mask.shape != (h, w):
            target_mask = cv2.resize(target_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_float = target_mask.astype(np.float32) / 255.0
        # Smooth mask edge slightly
        mask_smooth = cv2.GaussianBlur(mask_float, (9, 9), 0)
        # Apply soft anatomical gating
        gated_heatmap = heatmap_resized * (0.15 + 0.85 * mask_smooth)
    else:
        gated_heatmap = heatmap_resized

    color_map = apply_colormap_to_heatmap(gated_heatmap, (w, h))
    blended = cv2.addWeighted(orig_rgb, beta, color_map, alpha, 0)
    return Image.fromarray(blended)


def create_four_panel_grid(
    original_image: Image.Image,
    heatmap_raw: np.ndarray,
    rollout_raw: np.ndarray | None,
    model_name: str,
    finding_name: str,
    probability: float,
    anatomy: dict | None = None,
    bounding_boxes: list[dict] | None = None,
) -> dict:
    """Constructs the comprehensive 4-panel diagnostic visual explanation grid.

    Panels:
        - Panel 1: Original Chest Radiograph with Multi-Focal Bounding Boxes
        - Panel 2: Anatomical Segmentation Mask (Semantic multi-compartment overlay)
        - Panel 3: Model-Specific Attribution Heatmap (Grad-CAM / Grad-CAM++ / ViT Grad×Act / GAT Regional)
        - Panel 4: Fused Diagnostic Visual Overlay (Grounded to Anatomy with Bounding Boxes)

    Returns:
        dict containing base64 data URLs for panels and metadata.
    """
    w, h = original_image.size
    orig_rgb = original_image.convert("RGB")

    # 1. Segment anatomy if not provided
    if anatomy is None:
        anatomy = segment_chest_anatomy(original_image)

    # 2. Extract bounding boxes if not provided
    if bounding_boxes is None:
        bounding_boxes = extract_anatomical_bounding_boxes(
            heatmap=heatmap_raw,
            anatomy=anatomy,
            finding_name=finding_name,
            probability=probability,
            orig_size=(w, h),
        )

    orig_np = np.array(orig_rgb, dtype=np.uint8)

    from .data import enhance_radiograph_for_diagnostic_fidelity
    from .metrics import calculate_psnr
    enhanced_radiograph = enhance_radiograph_for_diagnostic_fidelity(original_image)
    psnr_result = calculate_psnr(orig_np, np.array(enhanced_radiograph))

    # Panel 1: Original Clean & Annotated Radiograph
    panel_1_clean_url = image_to_data_url(orig_rgb)
    annotated_img = draw_bounding_boxes_on_image(orig_rgb, bounding_boxes, psnr_db=psnr_result["psnr_db"])
    panel_1_annotated_url = image_to_data_url(annotated_img)

    # Panel 2: Semantic Multi-Compartment Anatomical Mask
    # Color-coded overlay (Blue = Lung Parenchyma, Red = Cardiac Silhouette, Green = Perihilar Zone)
    semantic_overlay = anatomy.get("semantic_overlay")
    if semantic_overlay is None:
        semantic_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    # Blend with original radiograph for translucent anatomical reference
    anatomy_fused = cv2.addWeighted(orig_np, 0.55, semantic_overlay, 0.45, 0)
    panel_2_img = Image.fromarray(anatomy_fused)
    panel_2_url = image_to_data_url(panel_2_img)

    # Binary pure lung mask (matching user reference Image 2)
    binary_lung_img = Image.fromarray(anatomy.get("lung_mask", np.zeros((h, w), dtype=np.uint8)))
    panel_2_binary_url = image_to_data_url(binary_lung_img)

    # Panel 3: Model-Specific Attribution Heatmap
    color_heatmap = apply_colormap_to_heatmap(heatmap_raw, (w, h))
    panel_3_img = Image.fromarray(color_heatmap)
    panel_3_url = image_to_data_url(panel_3_img)

    # Panel 4: Fused Diagnostic Visual Overlay
    # Select anatomical mask for gating
    finding_lower = finding_name.lower()
    if "cardi" in finding_lower:
        gating_mask = anatomy.get("cardiac_mask")
    elif "perihilar" in finding_lower or "edema" in finding_lower:
        gating_mask = cv2.bitwise_or(anatomy.get("lung_mask", np.zeros((h, w), dtype=np.uint8)),
                                     anatomy.get("perihilar_mask", np.zeros((h, w), dtype=np.uint8)))
    else:
        gating_mask = anatomy.get("lung_mask")

    panel_4_fused_raw = create_fused_overlay(original_image, heatmap_raw, target_mask=gating_mask, alpha=0.45, beta=0.55)
    # Draw bounding boxes and PSNR badge onto Panel 4 fused overlay as well
    panel_4_img = draw_bounding_boxes_on_image(panel_4_fused_raw, bounding_boxes, color=(255, 230, 0), psnr_db=psnr_result["psnr_db"])
    panel_4_url = image_to_data_url(panel_4_img)

    # ViT rollout or secondary edge view for optional reference
    if rollout_raw is not None:
        color_rollout = apply_colormap_to_heatmap(rollout_raw, (w, h))
        rollout_img = Image.fromarray(color_rollout)
        rollout_url = image_to_data_url(rollout_img)
        rollout_title = "ViT Attention Rollout (12 Layers)"
        rollout_available = True
    else:
        orig_gray = np.array(original_image.convert("L"))
        edges = cv2.Canny(orig_gray, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        rollout_img = Image.fromarray(edges_color)
        rollout_url = image_to_data_url(rollout_img)
        rollout_title = "Anatomical Edge Boundaries"
        rollout_available = False

    # Generate unified 4-panel diagnostic composite figure (1 row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), facecolor="#0d1117")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    # Panel 1: Original with Focal Bounding Boxes
    axes[0].imshow(np.array(annotated_img))
    box_count = len(bounding_boxes)
    box_info = f" ({box_count} Focal Lesion{'s' if box_count != 1 else ''})" if box_count > 0 else ""
    axes[0].set_title(f"Panel 1: Grounded Bounding Boxes{box_info}\nPSNR: {psnr_result['psnr_db']} dB", color="#e6edf3", fontsize=11, pad=8, fontweight="bold")

    # Panel 2: Anatomical Segmentation Mask
    axes[1].imshow(anatomy_fused)
    axes[1].set_title("Panel 2: Anatomical Segmentation Mask\nBlue: Lungs | Red: Heart | Green: Hila", color="#58a6ff", fontsize=11, pad=8, fontweight="bold")

    # Panel 3: Model-Specific Attribution Heatmap
    axes[2].imshow(color_heatmap)
    method_name = {
        "vit": "ViT Grad × Activation",
        "densenet": "DenseNet121 Grad-CAM",
        "capsnet": "CapsNet Grad-CAM++",
        "gat": "GAT Regional Attention",
    }.get(model_name.lower(), "Model Attribution")
    axes[2].set_title(f"Panel 3: {method_name}", color="#e6edf3", fontsize=11, pad=8, fontweight="bold")

    # Panel 4: Fused Diagnostic Visual Overlay
    axes[3].imshow(np.array(panel_4_img))
    axes[3].set_title(
        f"Panel 4: Anatomically Fused Overlay\n{finding_name} ({probability:.1%}) | PSNR: {psnr_result['psnr_db']} dB",
        color="#00d2ff",
        fontsize=11,
        pad=8,
        fontweight="bold",
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    composite_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    return {
        "panel_1_original": panel_1_clean_url,
        "panel_1_annotated": panel_1_annotated_url,
        "panel_2_anatomy": panel_2_url,
        "panel_2_binary_lung": panel_2_binary_url,
        "panel_2_heatmap": panel_3_url,
        "panel_3_rollout": rollout_url,
        "panel_3_title": rollout_title,
        "panel_3_available": rollout_available,
        "panel_4_fused": panel_4_url,
        "composite_grid": composite_url,
        "bounding_boxes": bounding_boxes,
        "finding": finding_name,
        "probability": float(probability),
        "psnr_db": psnr_result["psnr_db"],
        "psnr_mse": psnr_result["mse"],
        "psnr_mae": psnr_result["mae"],
        "psnr_ssim": psnr_result["ssim"],
        "psnr_quality": psnr_result["quality"],
        "standard_range": psnr_result["standard_range"],
    }


