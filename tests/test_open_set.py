from __future__ import annotations
import numpy as np
import pytest
import torch

from backend.config import (
    ADDITIONAL_THORACIC_FINDINGS,
    ALL_THORACIC_FINDINGS,
    CLINICAL_PROMPT_TEMPLATES,
    COMPLETE_ANATOMY,
    FINDINGS,
    OPEN_SET_CONFIG,
)
from backend.open_set import (
    CLINICAL_CONCEPT_ANCHORS,
    OpenSetAnalyzer,
)


def test_expanded_ontology_coverage():
    """Verify that the expanded thoracic ontology defines all 28+ findings properly."""
    assert len(ALL_THORACIC_FINDINGS) >= 28
    assert len(FINDINGS) == 13
    assert len(ADDITIONAL_THORACIC_FINDINGS) >= 15
    
    # Check that COMPLETE_ANATOMY has every finding
    for finding in ALL_THORACIC_FINDINGS:
        assert finding in COMPLETE_ANATOMY, f"Missing {finding} in COMPLETE_ANATOMY"
        info = COMPLETE_ANATOMY[finding]
        assert "category" in info
        assert "region" in info
        assert "visual_concept" in info
        assert "clinical_note" in info

    assert len(CLINICAL_PROMPT_TEMPLATES) >= 3


def test_clinical_concept_anchors():
    """Verify semantic anchor vectors are 256-d, normalized, and cover all findings."""
    assert len(CLINICAL_CONCEPT_ANCHORS) == len(ALL_THORACIC_FINDINGS)
    for finding, vec in CLINICAL_CONCEPT_ANCHORS.items():
        assert vec.shape == (256,)
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-3)


def test_free_energy_computation():
    """Verify Helmholtz free energy calculation handles tensors and arrays correctly."""
    analyzer = OpenSetAnalyzer(temperature=1.0)
    
    # Confident high logits -> lower Helmholtz free energy
    high_logits = np.array([4.0, 1.0, -1.0, -2.0, 0.0])
    energy_high = analyzer.compute_free_energy(high_logits)
    
    # Uniform/low logits -> higher Helmholtz free energy
    low_logits = np.array([-3.0, -3.0, -3.0, -3.0, -3.0])
    energy_low = analyzer.compute_free_energy(low_logits)
    
    assert energy_high < energy_low
    
    # Test torch tensor input
    tensor_logits = torch.tensor([4.0, 1.0, -1.0, -2.0, 0.0])
    energy_tensor = analyzer.compute_free_energy(tensor_logits)
    assert np.isclose(energy_high, energy_tensor, atol=1e-5)


def test_entropy_computation():
    """Verify binary multi-label Shannon entropy calculation."""
    analyzer = OpenSetAnalyzer()
    
    # High uncertainty (all near 0.5)
    uncertain_probs = np.full(13, 0.5)
    entropy_uncertain = analyzer.compute_entropy(uncertain_probs)
    
    # High confidence (near 0 or 1)
    confident_probs = np.array([0.95] + [0.02] * 12)
    entropy_confident = analyzer.compute_entropy(confident_probs)
    
    assert entropy_uncertain > entropy_confident
    assert np.isclose(entropy_uncertain, 1.0, atol=1e-2)


def test_project_visual_to_anchors():
    """Verify visual feature projection against the 28+ finding ontology."""
    analyzer = OpenSetAnalyzer()
    
    # Test with ViT-like 768-d features
    vit_features = np.random.randn(768).astype(np.float32)
    candidates = analyzer.project_visual_to_anchors(vit_features, top_k=5)
    
    assert len(candidates) == 5
    for c in candidates:
        assert "finding" in c
        assert "score" in c
        assert "is_open_set" in c
        assert "category" in c
        assert "region" in c
        assert 0.0 <= c["score"] <= 1.0
        
    # Check that candidates are sorted descending
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)

    # Test with DenseNet-like 1024-d torch tensor
    densenet_tensor = torch.randn(1024)
    candidates_dn = analyzer.project_visual_to_anchors(densenet_tensor, top_k=8)
    assert len(candidates_dn) == 8


def test_synthesize_primary_conclusion_confident():
    """Verify primary conclusion synthesis when a finding clearly exceeds threshold."""
    analyzer = OpenSetAnalyzer(min_confidence=0.35)
    
    probs = {f: 0.05 for f in FINDINGS}
    probs["Cardiomegaly"] = 0.88
    probs["Enlarged Cardiomediastinum"] = 0.55
    
    result = analyzer.synthesize_primary_conclusion(
        model_name="DenseNet121",
        finding_probabilities=probs,
        recommended_threshold=0.45,
    )
    
    assert result["status"] == "confirmed"
    assert result["should_abstain"] is False
    assert result["primary_finding"] == "Cardiomegaly"
    assert "Cardiomegaly" in result["primary_conclusion"]
    assert result["primary_confidence"] == 0.88
    assert result["abstention_reason"] is None
    assert len(result["alternatives"]) >= 1
    # Ranked alternatives must not include the primary conclusion
    alt_names = [a["finding"] for a in result["alternatives"]]
    assert "Cardiomegaly" not in alt_names
    assert "Enlarged Cardiomediastinum" in alt_names


def test_synthesize_primary_conclusion_abstention():
    """Verify principled abstention when all findings are below confidence threshold."""
    analyzer = OpenSetAnalyzer(min_confidence=0.35)
    
    # Sub-threshold low probabilities
    probs = {f: 0.10 for f in FINDINGS}
    probs["Pleural Effusion"] = 0.22
    probs["Atelectasis"] = 0.18
    
    result = analyzer.synthesize_primary_conclusion(
        model_name="ViT-B/16",
        finding_probabilities=probs,
        recommended_threshold=0.40,
    )
    
    assert result["status"] == "abstained"
    assert result["should_abstain"] is True
    assert result["primary_finding"] is None
    assert "No reliable prediction" in result["primary_conclusion"]
    assert result["primary_confidence"] == 0.22
    assert "did not meet the calibrated" in result["abstention_reason"]
    assert len(result["alternatives"]) > 0


def test_synthesize_primary_conclusion_ood_rejection():
    """Verify abstention when input is detected as Out-Of-Distribution."""
    analyzer = OpenSetAnalyzer()
    
    probs = {f: 0.70 for f in FINDINGS}  # Even if logits are high
    ood_status = {
        "is_ood": True,
        "reason": "Non-medical photograph or extreme contrast artifact detected",
    }
    
    result = analyzer.synthesize_primary_conclusion(
        model_name="CheXpertCapsNet",
        finding_probabilities=probs,
        recommended_threshold=0.40,
        ood_status=ood_status,
    )
    
    assert result["status"] == "abstained"
    assert result["should_abstain"] is True
    assert "No reliable prediction" in result["primary_conclusion"]
    assert "Out-of-Distribution" in result["abstention_reason"]
