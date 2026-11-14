from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from .config import FINDINGS


def calculate_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    aucs, auprcs = [], []
    for idx, label in enumerate(FINDINGS):
        truth = y_true[:, idx]
        prob = y_prob[:, idx]
        auc = float(roc_auc_score(truth, prob)) if len(np.unique(truth)) > 1 else np.nan
        ap = float(average_precision_score(truth, prob)) if truth.sum() > 0 else np.nan
        if np.isfinite(auc): aucs.append(auc)
        if np.isfinite(ap): auprcs.append(ap)
        rows.append({
            "finding": label,
            "auroc": auc,
            "auprc": ap,
            "accuracy": float(accuracy_score(truth, y_pred[:, idx])),
            "precision": float(precision_score(truth, y_pred[:, idx], zero_division=0)),
            "recall": float(recall_score(truth, y_pred[:, idx], zero_division=0)),
            "f1": float(f1_score(truth, y_pred[:, idx], zero_division=0)),
            "support_positive": int(truth.sum()),
        })
    summary = {
        "threshold": float(threshold),
        "validation_samples": int(y_true.shape[0]),
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "micro_accuracy": float(accuracy_score(y_true.ravel(), y_pred.ravel())),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return summary, pd.DataFrame(rows)
