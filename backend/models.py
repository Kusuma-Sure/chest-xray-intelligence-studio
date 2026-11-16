"""Compatibility shim re-exporting model definitions from architectures.py and gat.py."""
from __future__ import annotations
from .architectures import (
    CapsuleLayer,
    CapsNet,
    CheXpertDenseNet121,
    CheXpertViT,
    build_model,
    forward_logits,
    squash,
)
from .gat import GraphAttentionLayer, ImageGraphGAT

__all__ = [
    "squash",
    "CapsuleLayer",
    "CapsNet",
    "GraphAttentionLayer",
    "ImageGraphGAT",
    "CheXpertDenseNet121",
    "CheXpertViT",
    "build_model",
    "forward_logits",
]
