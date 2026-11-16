"""Evaluation Script for CheXpert Multi-Label Models.

Computes validation metrics:
- Macro & Micro AUROC
- Macro & Micro AUPRC
- Precision, Recall, F1-Score
- Exact match / subset accuracy
- Finding-wise breakdown table
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import forward_logits
from .config import CHECKPOINT_DIR, FINDINGS
from .data import CheXpertDataset, find_dataset_root
from .metrics import calculate_metrics
from .model_registry import load_trained_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
    threshold: float = 0.50,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Runs evaluation over a DataLoader and computes comprehensive multi-label metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = forward_logits(model, images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    summary, per_finding = calculate_metrics(y_true, y_prob, threshold=threshold)
    return summary, per_finding, y_true, y_prob


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CheXpert model")
    parser.add_argument("--model", required=True, choices=["vit", "densenet", "capsnet", "gat"])
    parser.add_argument("--data-root", required=True, help="Path to CheXpert-v1.0-small folder")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    root = find_dataset_root(args.data_root)
    valid_csv = root / "valid.csv"

    dataset = CheXpertDataset(
        csv_path=valid_csv,
        data_root=root,
        model_name=args.model,
        max_samples=args.max_samples,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loading trained model: {args.model}")
    model = load_trained_model(args.model)

    print(f"Evaluating {len(dataset)} validation samples...")
    summary, per_finding, _, _ = evaluate_model_on_loader(
        model=model,
        loader=loader,
        device=DEVICE,
        threshold=args.threshold,
    )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_finding.to_csv(out_path / f"{args.model}_metrics.csv", index=False)

    print("\n========== EVALUATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("\n========== PER-FINDING METRICS ==========")
    print(per_finding.to_string(index=False))


if __name__ == "__main__":
    main()
