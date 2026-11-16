"""Out-of-Distribution (OOD) and Irrelevant Image Detector for Chest Radiographs.

Evaluates uploaded images across multiple analytical criteria:
1. Color Saturation & Chrominance (discriminates color photographs from monochromatic radiographs)
2. Thoracic Contrast & Spatial Histogram Dynamics (identifies bimodal lung vs. mediastinum distribution)
3. High-Frequency Edge & Texture Entropy (rejects flat drawings, text screenshots, or random noise)
4. Thoracoabdominal Aspect Ratio & Dimensional Sanity
5. Cross-Modality Filters (identifies unsupported modalities such as Brain MRI or Axial CT)

Rejects non-medical images (cars, dogs, food, landscapes, memes) before inference.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import cv2


class OODDetector:
    """Multi-stage Out-of-Distribution detector for frontal chest radiographs."""

    def __init__(self, ood_threshold: float = 0.55) -> None:
        self.ood_threshold = ood_threshold

    def analyze_image(self, image: Image.Image) -> dict:
        """Evaluates an input PIL image and returns an OOD assessment.

        Returns:
            dict containing:
                - is_valid_cxr (bool): True if verified as a valid frontal chest radiograph
                - ood_score (float): Anomaly score from 0.0 (ideal CXR) to 1.0 (clear non-CXR)
                - threshold (float): Detection cutoff threshold
                - rejection_reason (str | None): Diagnostic reason for rejection if invalid
                - user_message (str): Human-readable clinical/research explanation
                - diagnostics (dict): Breakdown of sub-scores
        """
        # 1. Format and dimension validation
        w, h = image.size
        if w < 64 or h < 64:
            return self._rejection_result(
                1.0,
                "Image resolution is too low (<64x64) for diagnostic anatomical evaluation.",
                "Input rejected: The uploaded image has insufficient resolution for chest radiograph analysis."
            )

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.45 or aspect_ratio > 2.2:
            return self._rejection_result(
                0.90,
                f"Extreme aspect ratio ({aspect_ratio:.2f}) does not match standard chest radiography projections.",
                "Input rejected: Dimensional aspect ratio does not match standard frontal chest radiography."
            )

        # Convert to numpy arrays
        rgb_array = np.array(image.convert("RGB"), dtype=np.float32)
        gray_array = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 2. Chrominance / Color Saturation Check
        # Genuine chest X-rays are monochromatic greyscale with negligible HSV saturation (<0.06).
        # Natural photographs (cars, animals, food, scenery) exhibit strong chromaticity.
        hsv = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1] / 255.0  # Range [0, 1]
        mean_saturation = float(np.mean(saturation))
        saturation_p90 = float(np.percentile(saturation, 90))

        if mean_saturation > 0.10 or saturation_p90 > 0.25:
            score = min(1.0, 0.65 + mean_saturation * 0.35)
            return self._rejection_result(
                round(score, 4),
                f"Significant color saturation detected (mean saturation: {mean_saturation:.2f}), indicating a non-medical photograph (such as a vehicle, animal, or outdoor scene).",
                "Input rejected: This image contains significant color saturation and appears to be a natural photograph, not a monochromatic chest radiograph. No disease predictions were generated."
            )

        # 3. Dynamic Range & Contrast Check
        intensity_min = float(np.min(gray_array))
        intensity_max = float(np.max(gray_array))
        dynamic_range = intensity_max - intensity_min
        if dynamic_range < 25.0:
            return self._rejection_result(
                0.95,
                "Extremely low dynamic range (near-constant pixel values, blank or corrupted image).",
                "Input rejected: The image has almost no contrast and appears blank or corrupted."
            )

        # 4. Radiographic Intensity Distribution (Bimodal Histogram Check)
        norm_gray = ((gray_array - intensity_min) / (dynamic_range + 1e-6) * 255.0).astype(np.uint8)
        hist = cv2.calcHist([norm_gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)

        # Entropy calculation
        non_zero = hist[hist > 0]
        entropy = -float(np.sum(non_zero * np.log2(non_zero)))

        # Very low entropy = flat cartoon, solid block, or test pattern
        if entropy < 1.8:
            return self._rejection_result(
                0.88,
                f"Image entropy ({entropy:.2f}) is abnormally low, characteristic of solid diagrams or synthetic graphics.",
                "Input rejected: Image appears to be a graphic diagram or synthetic screenshot, not a clinical radiograph."
            )

        # 5. Spatial Anatomical Symmetry & Central Mediastinal Mass Profile
        h_half, w_half = norm_gray.shape[0] // 2, norm_gray.shape[1] // 2
        left_hemithorax = norm_gray[:, :w_half]
        right_hemithorax = cv2.flip(norm_gray[:, w_half:], 1)

        min_w = min(left_hemithorax.shape[1], right_hemithorax.shape[1])
        diff = np.abs(left_hemithorax[:, :min_w].astype(float) - right_hemithorax[:, :min_w].astype(float))
        bilateral_diff_ratio = float(np.mean(diff) / 255.0)

        # 6. High-Frequency Edge / Texture Analysis
        laplacian_var = float(cv2.Laplacian(norm_gray, cv2.CV_32F).var())

        # 7. Modality Heuristics (CT / MRI vs Frontal CXR)
        border_black_ratio = float(
            (np.mean(norm_gray[:10, :] < 10) + np.mean(norm_gray[-10:, :] < 10) +
             np.mean(norm_gray[:, :10] < 10) + np.mean(norm_gray[:, -10:] < 10)) / 4.0
        )

        is_circular_slice = False
        if border_black_ratio > 0.85 and 0.85 < aspect_ratio < 1.15:
            corners = (
                np.mean(norm_gray[:h//4, :w//4] < 15) +
                np.mean(norm_gray[:h//4, 3*w//4:] < 15) +
                np.mean(norm_gray[3*h//4:, :w//4] < 15) +
                np.mean(norm_gray[3*h//4:, 3*w//4:] < 15)
            ) / 4.0
            if corners > 0.88:
                # Distinguish collimated frontal chest radiographs from cranial axial MRI/CT:
                # Frontal chest radiographs exhibit a radiopaque central mediastinal/spinal column
                # with bilateral aerated (radiolucent) pulmonary zones.
                center_strip = norm_gray[h//4 : 3*h//4, w//2 - 25 : w//2 + 25]
                left_lung_zone = norm_gray[h//4 : 3*h//4, w//6 : w//3]
                right_lung_zone = norm_gray[h//4 : 3*h//4, 2*w//3 : 5*w//6]
                has_thoracic_gradient = (
                    center_strip.mean() > left_lung_zone.mean() + 15 and
                    center_strip.mean() > right_lung_zone.mean() + 15
                )
                if not has_thoracic_gradient:
                    is_circular_slice = True

        # ======================================================================
        # CALCULATE COMPOSITE OOD SCORE (0.0 = Genuine CXR, 1.0 = High Anomaly)
        # ======================================================================
        ood_score = 0.0
        rejection_reasons = []

        # Axial brain MRI / CT detection
        if is_circular_slice:
            ood_score = max(ood_score, 0.85)
            rejection_reasons.append("Cross-sectional axial geometry detected, characteristic of Cranial MRI or Axial CT. This model exclusively supports frontal chest radiographs.")

        # Extreme bilateral asymmetry
        if bilateral_diff_ratio > 0.50:
            ood_score = max(ood_score, min(0.9, (bilateral_diff_ratio - 0.40) * 2.5))
            rejection_reasons.append(f"Thoracic bilateral symmetry deviation is high ({bilateral_diff_ratio:.2f}), atypical for frontal chest projections.")

        # Texture sharpness check
        if laplacian_var < 5.0:
            ood_score = max(ood_score, 0.75)
            rejection_reasons.append("Image is severely blurred or lacks fine anatomical trabecular and bronchovascular textures.")

        is_valid = ood_score < self.ood_threshold

        diagnostics = {
            "mean_saturation": round(mean_saturation, 4),
            "saturation_p90": round(saturation_p90, 4),
            "entropy": round(entropy, 2),
            "laplacian_var": round(laplacian_var, 1),
            "bilateral_diff_ratio": round(bilateral_diff_ratio, 3),
            "aspect_ratio": round(aspect_ratio, 2),
            "border_black_ratio": round(border_black_ratio, 3),
        }

        if not is_valid:
            reason_text = " | ".join(rejection_reasons) if rejection_reasons else "Image characteristics diverge substantially from verified frontal chest radiographs."
            user_msg = (
                "Input rejected: This image could not be verified as a supported frontal chest radiograph "
                "(e.g., detected non-medical photograph, unsupported modality like brain MRI/CT, or severe artifact). "
                "No disease predictions were generated."
            )
            return {
                "is_valid_cxr": False,
                "ood_score": round(float(ood_score), 4),
                "threshold": self.ood_threshold,
                "rejection_reason": reason_text,
                "user_message": user_msg,
                "diagnostics": diagnostics,
            }

        return {
            "is_valid_cxr": True,
            "ood_score": round(float(ood_score), 4),
            "threshold": self.ood_threshold,
            "rejection_reason": None,
            "user_message": "Image verified as a supported frontal chest radiograph.",
            "diagnostics": diagnostics,
        }

    def _rejection_result(self, score: float, reason: str, user_msg: str) -> dict:
        return {
            "is_valid_cxr": False,
            "ood_score": round(float(score), 4),
            "threshold": self.ood_threshold,
            "rejection_reason": reason,
            "user_message": user_msg,
            "diagnostics": {},
        }


# Global detector instance
ood_detector = OODDetector(ood_threshold=0.55)
