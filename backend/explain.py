"""Explainability compatibility layer forwarding to explainability.py and visualization.py."""
from __future__ import annotations
from PIL import Image
import torch
from .explainability import generate_visual_explanation
from .visualization import create_four_panel_grid, create_fused_overlay, image_to_data_url


def explain_prediction(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original: Image.Image,
    target_index: int,
) -> dict:
    """Legacy-compatible visual attribution wrapper returning overlays and 4-panel data."""
    raw = generate_visual_explanation(
        model=model,
        model_name=model_name,
        input_tensor=input_tensor,
        original_image=original,
        target_class_idx=target_index,
    )
    if not raw.get("available"):
        return raw

    # Create overlays and 4-panel grid
    grid = create_four_panel_grid(
        original_image=original,
        heatmap_raw=raw["heatmap_raw"],
        rollout_raw=raw.get("rollout_raw"),
        model_name=model_name,
        finding_name=raw.get("target_finding", "Finding"),
        probability=0.0,
    )

    return {
        "available": True,
        "method": raw.get("method"),
        "target_finding": raw.get("target_finding"),
        "target_index": target_index,
        "overlay": grid["panel_4_fused"],
        "panels": grid,
    }
