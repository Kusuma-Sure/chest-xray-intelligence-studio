"""Unit tests for training data pipelines, metrics, and patient separation."""
import numpy as np
import torch
from backend.config import FINDINGS
from backend.data import extract_patient_id, make_transform
from backend.metrics import calculate_metrics
from backend.train import compute_masked_loss


def test_extract_patient_id():
    path_1 = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    path_2 = "chexpert/valid/patient64541/study2/view2_lateral.jpg"
    assert extract_patient_id(path_1) == "patient00001"
    assert extract_patient_id(path_2) == "patient64541"


def test_masked_loss_calculation():
    logits = torch.zeros(2, 13)
    # Target with some positive (1), negative (0), and uncertain (-1)
    labels = torch.tensor([
        [1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    ])
    pos_weight = torch.ones(13)

    loss_masked = compute_masked_loss(logits, labels, pos_weight, policy="masked")
    assert not torch.isnan(loss_masked)
    assert loss_masked.item() > 0.0


def test_metrics_calculation_complete():
    # 4 synthetic samples across 13 classes
    y_true = np.array([
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    ])
    y_prob = np.array([
        [0.8, 0.2, 0.7, 0.1, 0.3, 0.9, 0.2, 0.1, 0.85, 0.3, 0.2, 0.75, 0.1],
        [0.2, 0.85, 0.1, 0.9, 0.2, 0.1, 0.8, 0.2, 0.1, 0.7, 0.3, 0.2, 0.8],
        [0.75, 0.6, 0.2, 0.1, 0.8, 0.3, 0.1, 0.65, 0.3, 0.1, 0.7, 0.2, 0.3],
        [0.1, 0.3, 0.8, 0.7, 0.2, 0.65, 0.4, 0.2, 0.9, 0.3, 0.1, 0.85, 0.75],
    ])

    summary, per_finding = calculate_metrics(y_true, y_prob, threshold=0.50)
    assert summary["validation_samples"] == 4
    assert len(per_finding) == 13
    assert "macro_auroc" in summary
    assert "macro_f1" in summary
    assert "micro_f1" in summary
