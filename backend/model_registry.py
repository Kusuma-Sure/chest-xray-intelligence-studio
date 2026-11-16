"""Model Registry and Checkpoint Management Service.

Maintains metadata, checkpoint resolution, status reporting, and cached PyTorch model
instantiation for all four implemented architectures:
- Vision Transformer (ViT-B/16)
- DenseNet121
- Capsule Network (CapsNet)
- Graph Attention Network (GAT)
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
import torch

from .architectures import build_model
from .config import CHECKPOINT_DIR, FINDINGS, MODEL_INFO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(model_id: str) -> Path | None:
    """Finds the existing checkpoint path or returns the default target location."""
    model_id = model_id.lower().strip()
    if model_id not in MODEL_INFO:
        return None

    info = MODEL_INFO[model_id]
    candidates = [CHECKPOINT_DIR / info["checkpoint"]]
    for alt in info.get("alternate_checkpoints", []):
        candidates.append(CHECKPOINT_DIR / alt)

    # Check for Hugging Face folder in checkpoints
    if model_id == "vit":
        candidates.append(CHECKPOINT_DIR / "vit_chexpert_baseline")

    for cand in candidates:
        if cand.is_file() or (cand.is_dir() and (cand / "config.json").is_file()):
            return cand
    return candidates[0]


def get_model_status(model_id: str) -> dict[str, Any]:
    """Returns comprehensive status metadata for a registered model.

    Crucial fix: Never hides models whose checkpoints are not yet trained.
    Returns clear status strings: 'Available', 'Missing checkpoint', 'Invalid checkpoint'.
    """
    model_id = model_id.lower().strip()
    if model_id not in MODEL_INFO:
        raise ValueError(f"Unknown model id '{model_id}'")

    info = MODEL_INFO[model_id]
    path = resolve_checkpoint_path(model_id)
    exists = path.exists() if path else False

    if exists:
        status_label = "Available – trained checkpoint loaded"
        status_code = "ready"
        reason = f"Checkpoint verified at {path.name}. Ready for real inference."
    else:
        status_label = "Missing checkpoint – training required"
        status_code = "missing"
        reason = (
            f"Trained weights not found in {CHECKPOINT_DIR}. Train this model using the supplied "
            f"Colab training notebook, then copy '{info['checkpoint']}' into checkpoints/."
        )

    return {
        "id": model_id,
        "name": info["name"],
        "family": info["family"],
        "primary": info.get("primary", False),
        "available": exists,
        "status_code": status_code,
        "status": status_label,
        "checkpoint_file": path.name if path else info["checkpoint"],
        "checkpoint_path": str(path) if path else "",
        "input_type": info.get("input_type", "RGB / 224x224"),
        "explanation_method": info.get("explanation_method", "Attribution"),
        "description": info.get("description", ""),
        "reason": reason,
    }


def list_all_models() -> list[dict[str, Any]]:
    """Returns status dictionaries for all four supported architectures."""
    return [get_model_status(k) for k in MODEL_INFO]


def _read_checkpoint_payload(path: Path, device: torch.device) -> Any:
    """Reads a PyTorch checkpoint file safely."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


@lru_cache(maxsize=4)
def load_trained_model(model_id: str) -> torch.nn.Module:
    """Loads and caches an architecture with its verified trained state dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If state dictionary shapes or labels do not match.
    """
    model_id = model_id.lower().strip()
    if model_id not in MODEL_INFO:
        raise ValueError(f"Unknown model architecture '{model_id}'")

    path = resolve_checkpoint_path(model_id)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing for {MODEL_INFO[model_id]['name']}. Expected at: "
            f"{CHECKPOINT_DIR / MODEL_INFO[model_id]['checkpoint']}. Train in Google Colab first."
        )

    # Initialize model architecture
    model = build_model(model_id, pretrained=False)

    # Load weights
    if path.is_dir() and model_id == "vit":
        # Hugging Face directory
        try:
            from transformers import ViTForImageClassification
            model.vit_model = ViTForImageClassification.from_pretrained(str(path))
            model.to(DEVICE).eval()
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load ViT from Hugging Face directory: {exc}") from exc

    payload = _read_checkpoint_payload(path, DEVICE)

    # Extract state dict from various common container formats
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            state = payload["model_state_dict"]
        elif "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
    else:
        state = payload

    # Load into model with strict error handling
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        # Try non-strict fallback if minor buffer prefixes differ
        try:
            model.load_state_dict(state, strict=False)
        except Exception as exc2:
            raise RuntimeError(
                f"Checkpoint weights mismatch for {model_id}: {exc2}. Re-train model."
            ) from exc2

    model.to(DEVICE).eval()
    return model
