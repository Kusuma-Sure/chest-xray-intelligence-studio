"""Anatomical Segmentation and Lesion Localization Module for Chest Radiographs.

Implements:
1. Bilateral Lung Field Segmentation (Left and Right lung masks separating parenchyma from background, spine, and ribs)
2. Cardiac Silhouette Segmentation (Cardiothoracic boundary isolation in the lower-middle mediastinum)
3. Perihilar Zone Isolation (Central bronchovascular roots)
4. Color-coded Semantic Multi-Compartment Overlay (Blue = Lungs, Red = Cardiac, Green = Perihilar)
5. Anatomically Grounded Multi-Focal Bounding Box Extraction with Clinical Zone Labeling:
   - Right Upper Lobe (RUL), Right Mid Zone (RML), Right Lower Lobe (RLL)
   - Left Upper Lobe (LUL), Left Mid Zone, Left Lower Lobe (LLL) / Retrocardiac
   - Cardiac Silhouette / Transverse Cardiac Diameter
6. Crisp Visual Bounding Box Annotation on Radiographs
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw


def segment_chest_anatomy(image: Image.Image) -> dict:
    """Segments anatomical thoracic compartments from a chest radiograph.

    Returns:
        dict containing:
            - lung_mask: (H, W) uint8 {0, 255} binary bilateral lung field mask
            - left_lung_mask: (H, W) uint8 {0, 255} left lung mask (patient left = image right)
            - right_lung_mask: (H, W) uint8 {0, 255} right lung mask (patient right = image left)
            - cardiac_mask: (H, W) uint8 {0, 255} cardiac silhouette mask
            - perihilar_mask: (H, W) uint8 {0, 255} bilateral perihilar zones mask
            - semantic_overlay: (H, W, 3) RGB uint8 multi-compartment overlay
    """
    w, h = image.size
    img_gray = np.array(image.convert("L"))

    # Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    # Gaussian smoothing to remove high-frequency rib noise
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # Adaptive thoracic thresholding
    valid_body_mask = blurred > 15
    valid_pixels = blurred[valid_body_mask]
    if len(valid_pixels) > 0:
        thresh_val = np.percentile(valid_pixels, 42)
    else:
        thresh_val = 120

    lung_raw = (blurred < thresh_val) & valid_body_mask
    lung_uint8 = np.uint8(lung_raw * 255)

    # Morphological opening (remove stray pixels) and closing (fill vascular gaps)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    lung_cleaned = cv2.morphologyEx(lung_uint8, cv2.MORPH_OPEN, kernel_small)
    lung_cleaned = cv2.morphologyEx(lung_cleaned, cv2.MORPH_CLOSE, kernel_large)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lung_cleaned)

    lung_mask = np.zeros((h, w), dtype=np.uint8)
    left_lung_mask = np.zeros((h, w), dtype=np.uint8)
    right_lung_mask = np.zeros((h, w), dtype=np.uint8)

    mid_x = w // 2
    candidates = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        cy = centroids[lbl][1]
        cx = centroids[lbl][0]
        if area > (w * h * 0.02) and (0.10 * h < cy < 0.90 * h) and (0.05 * w < cx < 0.95 * w):
            candidates.append((area, lbl, cx, cy))

    candidates.sort(key=lambda x: x[0], reverse=True)

    if len(candidates) >= 2:
        for c in candidates[:2]:
            lbl = c[1]
            cx = centroids[lbl][0]
            mask_part = (labels == lbl).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                hull = cv2.convexHull(contours[0])
                filled_part = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(filled_part, [hull], -1, 255, -1)
                lung_mask = cv2.bitwise_or(lung_mask, filled_part)
                if cx < mid_x:
                    right_lung_mask = cv2.bitwise_or(right_lung_mask, filled_part)
                else:
                    left_lung_mask = cv2.bitwise_or(left_lung_mask, filled_part)
    elif len(candidates) == 1:
        lbl = candidates[0][1]
        cx = centroids[lbl][0]
        mask_part = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(contours[0])
            filled_part = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled_part, [hull], -1, 255, -1)
            lung_mask = filled_part
            if cx < mid_x:
                right_lung_mask = filled_part
            else:
                left_lung_mask = filled_part
    else:
        # Standard thoracic anatomical fallback
        cv2.ellipse(lung_mask, (int(w * 0.32), int(h * 0.48)), (int(w * 0.16), int(h * 0.32)), 0, 0, 360, 255, -1)
        cv2.ellipse(lung_mask, (int(w * 0.68), int(h * 0.48)), (int(w * 0.16), int(h * 0.32)), 0, 0, 360, 255, -1)
        cv2.ellipse(right_lung_mask, (int(w * 0.32), int(h * 0.48)), (int(w * 0.16), int(h * 0.32)), 0, 0, 360, 255, -1)
        cv2.ellipse(left_lung_mask, (int(w * 0.68), int(h * 0.48)), (int(w * 0.16), int(h * 0.32)), 0, 0, 360, 255, -1)

    # 2. Cardiac Silhouette Masking (Between lungs, lower-mid mediastinum, apex leaning patient left)
    cardiac_mask = np.zeros((h, w), dtype=np.uint8)
    card_center_x = int(w * 0.52)
    card_center_y = int(h * 0.62)
    card_axis_x = int(w * 0.18)
    card_axis_y = int(h * 0.18)
    cv2.ellipse(cardiac_mask, (card_center_x, card_center_y), (card_axis_x, card_axis_y), 15, 0, 360, 255, -1)

    # 3. Perihilar Zone Masking (Central bilateral hilar roots)
    perihilar_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(perihilar_mask, (int(w * 0.42), int(h * 0.46)), (int(w * 0.08), int(h * 0.12)), 0, 0, 360, 255, -1)
    cv2.ellipse(perihilar_mask, (int(w * 0.58), int(h * 0.46)), (int(w * 0.08), int(h * 0.12)), 0, 0, 360, 255, -1)

    # 4. Color-coded Semantic Multi-Compartment Overlay
    # Blue: Lung Parenchyma / Alveolar Space (RGB: [0, 140, 255])
    # Red: Cardiac Silhouette (RGB: [230, 40, 40])
    # Green: Perihilar Zones (RGB: [40, 210, 80])
    semantic_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    semantic_overlay[lung_mask > 0] = [0, 140, 255]
    semantic_overlay[cardiac_mask > 0] = [230, 40, 40]
    semantic_overlay[perihilar_mask > 0] = [40, 210, 80]

    return {
        "lung_mask": lung_mask,
        "left_lung_mask": left_lung_mask,
        "right_lung_mask": right_lung_mask,
        "cardiac_mask": cardiac_mask,
        "perihilar_mask": perihilar_mask,
        "semantic_overlay": semantic_overlay,
        "is_heuristic": True,
        "method": "Morphological CLAHE & Connected-Component Approximation",
        "disclaimer": "Experimental Anatomical Approximation (Heuristic) — Not a clinically validated segmentation network.",
    }


def extract_anatomical_bounding_boxes(
    heatmap: np.ndarray,
    anatomy: dict,
    finding_name: str,
    probability: float,
    orig_size: tuple[int, int],
) -> list[dict]:
    """Detects and localizes focal pathological bounding boxes within anatomical zones.

    Args:
        heatmap: 2D normalized attribution map [0.0, 1.0]
        anatomy: Dictionary returned by segment_chest_anatomy
        finding_name: Name of finding (e.g. 'Lung Opacity', 'Cardiomegaly')
        probability: Model predicted probability [0.0, 1.0]
        orig_size: (width, height) of original radiograph

    Returns:
        List of bounding box dicts containing:
            - box: [xmin, ymin, xmax, ymax]
            - label: Finding label string with percentage
            - zone: Anatomical subregion string (e.g. 'Right Lower Lobe (RLL)')
            - area_pct: Percentage of total image area
            - peak_norm: Relative peak attribution [0.0, 1.0]
    """
    w, h = orig_size
    resized_cam = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    resized_cam = np.clip(resized_cam, 0.0, 1.0)

    finding_lower = finding_name.lower()
    if "cardi" in finding_lower:
        target_mask = anatomy["cardiac_mask"]
    elif "perihilar" in finding_lower or "edema" in finding_lower:
        target_mask = cv2.bitwise_or(anatomy["lung_mask"], anatomy["perihilar_mask"])
    else:
        target_mask = anatomy["lung_mask"]

    # Soft anatomical masking and re-normalization
    masked_cam = resized_cam * (target_mask.astype(np.float32) / 255.0)
    max_in_mask = float(np.max(masked_cam))

    if max_in_mask < 0.05:
        return []

    # Re-normalize to highlight focal peaks strictly inside the anatomical structure
    norm_masked = masked_cam / max_in_mask

    # Threshold for focal lesion detection (at least 45% of peak anatomical activation)
    act_thresh = 0.45
    binary_cam = np.uint8((norm_masked >= act_thresh) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_cam = cv2.morphologyEx(binary_cam, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cam)
    mid_x = w // 2
    min_area = w * h * 0.003  # At least 0.3% of image area

    boxes = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        bx = int(stats[lbl, cv2.CC_STAT_LEFT])
        by = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[lbl]

        # Anatomical zone identification
        if "cardi" in finding_lower:
            zone = "Cardiac Silhouette / Transverse Diameter"
        elif cx < mid_x:
            if cy < 0.38 * h:
                zone = "Right Upper Lobe (RUL)"
            elif cy < 0.65 * h:
                zone = "Right Mid Zone (RML)"
            else:
                zone = "Right Lower Lobe (RLL)"
        else:
            if cy < 0.42 * h:
                zone = "Left Upper Lobe (LUL)"
            elif cy < 0.68 * h:
                zone = "Left Mid Zone"
            else:
                zone = "Left Lower Lobe (LLL)"

        boxes.append({
            "box": [bx, by, bx + bw, by + bh],
            "label": f"{finding_name} ({probability:.1%})",
            "zone": zone,
            "area_pct": round(area / (w * h) * 100, 1),
            "peak_norm": round(float(np.max(norm_masked[by:by+bh, bx:bx+bw])), 2),
        })

    # Sort descending by peak activation
    boxes.sort(key=lambda b: b["peak_norm"], reverse=True)

    # If no connected component passed the threshold but probability is high (>= 0.40),
    # generate a localized anatomical bounding box around the active lobe
    if not boxes and probability >= 0.40:
        y_indices, x_indices = np.where(masked_cam > 0.15)
        if len(x_indices) > 0 and len(y_indices) > 0:
            bx1, bx2 = int(np.min(x_indices)), int(np.max(x_indices))
            by1, by2 = int(np.min(y_indices)), int(np.max(y_indices))
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            zone = "Bilateral Lung Fields" if abs(cx - mid_x) < w * 0.1 else ("Right Lung" if cx < mid_x else "Left Lung")
            boxes.append({
                "box": [bx1, by1, bx2, by2],
                "label": f"{finding_name} ({probability:.1%})",
                "zone": zone,
                "area_pct": round((bx2 - bx1) * (by2 - by1) / (w * h) * 100, 1),
                "peak_norm": round(max_in_mask, 2),
            })

    return boxes


def draw_bounding_boxes_on_image(
    image: Image.Image,
    boxes: list[dict],
    color: tuple[int, int, int] = (255, 60, 60),
    line_width: int = 2,
    psnr_db: float | None = None,
) -> Image.Image:
    """Renders crisp bounding boxes with label badges and PSNR directly onto the radiograph."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for b in boxes:
        bx1, by1, bx2, by2 = b["box"]
        label_text = f"{b['label']} | {b['zone']}"

        # Draw bounding rectangle
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=line_width)

        # Draw filled label badge
        badge_w = min(len(label_text) * 7 + 8, w - bx1)
        badge_h = 16
        badge_y = max(0, by1 - badge_h)

        draw.rectangle([bx1, badge_y, bx1 + badge_w, badge_y + badge_h], fill=color)
        draw.text((bx1 + 4, badge_y + 2), label_text, fill=(255, 255, 255))

    # Render prominent PSNR badge directly on the presented radiograph
    if psnr_db is not None:
        psnr_tag = "Near-Lossless" if psnr_db >= 40.0 else ("Standard" if psnr_db >= 30.0 else "Compressed")
        psnr_str = f"PSNR: {psnr_db:.1f} dB ({psnr_tag})"
        badge_w = min(len(psnr_str) * 7 + 14, w - 16)
        badge_h = 20
        px1 = max(0, w - badge_w - 8)
        py1 = max(0, h - badge_h - 8)
        px2 = w - 8
        py2 = h - 8
        draw.rectangle([px1, py1, px2, py2], fill=(13, 17, 23), outline=(0, 210, 255), width=1)
        draw.text((px1 + 6, py1 + 4), psnr_str, fill=(0, 210, 255))

    return annotated
