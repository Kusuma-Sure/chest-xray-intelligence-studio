"""Multi-Model Prediction and Inference Orchestrator.

Handles:
1. Out-of-Distribution (OOD) / Irrelevant Image Validation
2. Multi-Model Selection and Dynamic Loading
3. Preprocessing (Model-specific Color / Resolution Normalization)
4. Multi-Label Sigmoid Inference across 13 CheXpert Findings
5. Batch Inference (1 to 5 Simultaneous Images)
6. 4-Panel XAI Visual Explanation Integration
7. Evidence-Grounded GraphRAG Retrieval
"""
from __future__ import annotations
import time
from typing import Any
import numpy as np
import torch
from PIL import Image

from .architectures import forward_logits
from .config import DEFAULT_THRESHOLD, FINDINGS, MODEL_INFO
from .data import make_transform
from .explainability import generate_visual_explanation
from .graph_rag import retrieve_graph_context
from .model_registry import DEVICE, get_model_status, load_trained_model
from .ood_detector import ood_detector
from .visualization import create_four_panel_grid


class MedicalPredictor:
    """Core prediction service for Chest X-ray Intelligence Studio."""

    def __init__(self) -> None:
        self.device = DEVICE

    def predict_single_image(
        self,
        image: Image.Image,
        model_name: str = "vit",
        threshold: float = DEFAULT_THRESHOLD,
        target_finding_idx: int | None = None,
        explain: bool = True,
    ) -> dict[str, Any]:
        """Runs the complete inference, XAI, and GraphRAG journey for one radiograph.

        Args:
            image: Uploaded PIL Image
            model_name: One of 'vit', 'densenet', 'capsnet', 'gat'
            threshold: Multi-label classification threshold [0.0, 1.0]
            target_finding_idx: Optional specific finding index to explain
            explain: Whether to generate 4-panel visual explanation
        """
        model_name = model_name.lower().strip()
        if model_name not in MODEL_INFO:
            raise ValueError(f"Unknown model architecture '{model_name}'.")

        started_time = time.perf_counter()

        # Step 1: Out-of-Distribution (OOD) & Domain Verification
        ood_result = ood_detector.analyze_image(image)
        if not ood_result["is_valid_cxr"]:
            return {
                "success": False,
                "is_ood": True,
                "ood_details": ood_result,
                "model_name": MODEL_INFO[model_name]["name"],
                "model_id": model_name,
                "user_message": ood_result["user_message"],
                "rejection_reason": ood_result["rejection_reason"],
                "processing_time_ms": round((time.perf_counter() - started_time) * 1000, 2),
                "findings": [],
                "positive_findings": [],
                "disclaimer": "No disease predictions were generated for non-verified or out-of-distribution input.",
            }

        # Step 2: Check Checkpoint Availability
        model_status = get_model_status(model_name)
        if not model_status["available"]:
            return {
                "success": False,
                "is_ood": False,
                "model_unavailable": True,
                "model_name": model_status["name"],
                "model_id": model_name,
                "status": model_status["status"],
                "reason": model_status["reason"],
                "user_message": (
                    f"Inference disabled: {model_status['name']} does not have a trained checkpoint installed. "
                    "In adherence to research integrity, mock predictions are strictly prohibited. "
                    f"Please train the architecture in Google Colab and place '{model_status['checkpoint_file']}' "
                    "into the checkpoints/ directory."
                ),
                "processing_time_ms": round((time.perf_counter() - started_time) * 1000, 2),
                "findings": [],
                "positive_findings": [],
            }

        # Step 3: Load Model & Preprocess Input
        model = load_trained_model(model_name)
        transform = make_transform(model_name, train=False)

        # Convert to expected color mode: CapsNet uses Grayscale 'L', others use 'RGB'
        mode = "L" if model_name == "capsnet" else "RGB"
        prep_img = image.convert(mode)
        input_tensor = transform(prep_img).unsqueeze(0).to(self.device)

        # Step 4: Multi-label Sigmoid Prediction
        try:
            if explain:
                # Keep autograd available for Grad-CAM / Gradient x Activation
                logits = forward_logits(model, input_tensor)
                probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
            else:
                with torch.no_grad():
                    logits = forward_logits(model, input_tensor)
                    probabilities = torch.sigmoid(logits)[0].cpu().numpy()
        except Exception as exc:
            raise RuntimeError(f"Model forward pass failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started_time) * 1000

        # Step 5: Format 13 Finding Scores
        findings_rows = []
        positive_findings = []
        for idx, (label, prob) in enumerate(zip(FINDINGS, probabilities)):
            is_pos = bool(prob >= threshold)
            item = {
                "index": idx,
                "finding": label,
                "probability": float(prob),
                "positive": is_pos,
                "threshold": float(threshold),
            }
            findings_rows.append(item)
            if is_pos:
                positive_findings.append(item)

        # Determine target finding for explainability focus
        if target_finding_idx is not None and 0 <= target_finding_idx < len(FINDINGS):
            focus_idx = target_finding_idx
        else:
            focus_idx = int(np.argmax(probabilities))

        focus_finding = FINDINGS[focus_idx]
        focus_prob = float(probabilities[focus_idx])

        # Step 6: Generate Explainability (4-Panel Visual Grid)
        xai_data = {"available": False, "reason": "Explainability was not requested."}
        if explain:
            try:
                raw_xai = generate_visual_explanation(
                    model=model,
                    model_name=model_name,
                    input_tensor=input_tensor,
                    original_image=image,
                    target_class_idx=focus_idx,
                )
                if raw_xai.get("available"):
                    four_panels = create_four_panel_grid(
                        original_image=image,
                        heatmap_raw=raw_xai["heatmap_raw"],
                        rollout_raw=raw_xai.get("rollout_raw"),
                        model_name=model_name,
                        finding_name=focus_finding,
                        probability=focus_prob,
                    )
                    xai_data = {
                        "available": True,
                        "method": raw_xai["method"],
                        "target_finding": focus_finding,
                        "target_class_idx": focus_idx,
                        "target_probability": focus_prob,
                        "panels": four_panels,
                    }
                else:
                    xai_data = raw_xai
            except Exception as exc:
                xai_data = {"available": False, "reason": f"Visual attribution failed: {exc}"}

        # Step 7: GraphRAG Context & Strict Factual Grounding
        graph_rag_data = retrieve_graph_context(
            model_name=MODEL_INFO[model_name]["name"],
            positive_findings=positive_findings,
            target_finding=focus_finding,
            xai_method=xai_data.get("method", "Attribution"),
            threshold=threshold,
        )

        return {
            "success": True,
            "is_ood": False,
            "model_id": model_name,
            "model_name": MODEL_INFO[model_name]["name"],
            "threshold": float(threshold),
            "processing_time_ms": round(elapsed_ms, 2),
            "input_resolution": f"{image.width} × {image.height}",
            "findings": findings_rows,
            "positive_findings": positive_findings,
            "highest_finding": focus_finding,
            "highest_probability": focus_prob,
            "explanation": xai_data,
            "graph_rag": graph_rag_data,
            "disclaimer": (
                "Academic Research Prototype: Predictions and attribution maps are probabilistic "
                "deep learning outputs for educational and scientific inquiry. Not approved as a medical device."
            ),
        }

    def predict_batch(
        self,
        images: list[Image.Image],
        model_name: str = "vit",
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Processes 1 to 5 images sequentially, returning structured results per image."""
        if len(images) > 5:
            images = images[:5]

        results = []
        for idx, img in enumerate(images):
            res = self.predict_single_image(
                image=img,
                model_name=model_name,
                threshold=threshold,
                explain=True,
            )
            res["image_index"] = idx
            results.append(res)
        return results


# Global predictor instance
predictor = MedicalPredictor()
