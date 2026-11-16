"""Unit tests for all four deep learning architectures."""
import pytest
import torch
from backend.config import FINDINGS
from backend.architectures import (
    build_model,
    forward_logits,
    squash,
    CapsNet,
    CheXpertDenseNet121,
    CheXpertViT,
)
from backend.gat import ImageGraphGAT


def test_squash_function():
    # Test that vector squashing bounds lengths strictly between [0, 1)
    x = torch.randn(2, 10, 16) * 100.0
    squashed = squash(x, dim=-1)
    lengths = torch.norm(squashed, dim=-1)
    assert torch.all(lengths < 1.0)
    assert torch.all(lengths >= 0.0)


def test_densenet121_output_shape():
    model = CheXpertDenseNet121(num_classes=len(FINDINGS), pretrained=False).eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = forward_logits(model, x)
    assert tuple(out.shape) == (2, len(FINDINGS))


def test_capsnet_dynamic_routing_output_shape():
    model = CapsNet(num_classes=len(FINDINGS), routing_iterations=2).eval()
    x_gray = torch.zeros(2, 1, 224, 224)
    with torch.no_grad():
        out = forward_logits(model, x_gray)
    assert tuple(out.shape) == (2, len(FINDINGS))


def test_capsnet_dictionary_forward():
    model = CapsNet(num_classes=len(FINDINGS), routing_iterations=2).eval()
    x_gray = torch.zeros(1, 1, 224, 224)
    with torch.no_grad():
        out = model(x_gray, return_dict=True)
    assert isinstance(out, dict)
    assert "logits" in out
    assert "probabilities" in out
    assert "capsules" in out
    assert tuple(out["probabilities"].shape) == (1, len(FINDINGS))
    assert tuple(out["capsules"].shape) == (1, len(FINDINGS), 16)


def test_gat_output_shape_and_attention():
    model = ImageGraphGAT(num_classes=len(FINDINGS), pretrained=False, grid_size=7).eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = forward_logits(model, x)
    assert tuple(out.shape) == (2, len(FINDINGS))

    region_map = model.get_region_importance()
    assert region_map is not None
    assert tuple(region_map.shape) == (7, 7)


def test_build_model_factory():
    for name in ["vit", "densenet", "capsnet", "gat"]:
        m = build_model(name, pretrained=False)
        assert m is not None

    with pytest.raises(ValueError):
        build_model("unknown_model_xyz")
