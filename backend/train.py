"""Unified Model Training and Validation Pipeline for CheXpert Multi-Label Classification.

Supports all four model architectures:
- ViT-B/16 (Hugging Face Transformers / primary model)
- DenseNet121
- CapsNet (with dynamic routing and vector capsules)
- ImageGraphGAT (spatial graph reasoning)

Features:
- Configurable uncertainty policies (-1 mapped to 0, 1, or masked out of loss)
- Patient-level split protection
- Mixed precision support where available
- Checkpoint persistence and history logging
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import build_model, forward_logits
from .config import CHECKPOINT_DIR, FINDINGS
from .data import CheXpertDataset, compute_pos_weight, find_dataset_root
from .metrics import calculate_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    policy: str = "zero",
) -> torch.Tensor:
    """Computes BCE loss handling CheXpert uncertainty policies."""
    if policy == "masked":
        # -1 marks uncertain labels to be excluded from gradient calculation
        mask = (labels >= 0.0).float()
        clamped_labels = torch.clamp(labels, min=0.0, max=1.0)
        unreduced_loss = F.binary_cross_entropy_with_logits(
            logits, clamped_labels, pos_weight=pos_weight, reduction="none"
        )
        return (unreduced_loss * mask).sum() / (mask.sum() + 1e-7)
    else:
        # For 'zero' or 'ones', labels are already binary [0, 1]
        clamped = torch.clamp(labels, min=0.0, max=1.0)
        return F.binary_cross_entropy_with_logits(logits, clamped, pos_weight=pos_weight)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    policy: str = "zero",
    threshold: float = 0.50,
) -> tuple[float, dict, Any, np.ndarray, np.ndarray]:
    """Runs a validation pass and returns loss, summary dictionary, and predictions."""
    model.eval()
    losses = []
    probs = []
    truths = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = forward_logits(model, images)
            loss = compute_masked_loss(logits, labels, pos_weight, policy)
            losses.append(float(loss.item()) * images.size(0))

            probs.append(torch.sigmoid(logits).cpu().numpy())
            # For metric calculation, treat negative/uncertain as 0
            binary_labels = torch.clamp(labels, 0.0, 1.0).cpu().numpy()
            truths.append(binary_labels)

    y_prob = np.concatenate(probs, axis=0)
    y_true = np.concatenate(truths, axis=0)
    val_loss = float(sum(losses) / len(loader.dataset))
    summary, per_finding = calculate_metrics(y_true, y_prob, threshold=threshold)
    return val_loss, summary, per_finding, y_true, y_prob


def main():
    parser = argparse.ArgumentParser(description="Train CheXpert multi-label radiograph models")
    parser.add_argument("--model", required=True, choices=["vit", "densenet", "capsnet", "gat"])
    parser.add_argument("--data-root", required=True, help="Root path of CheXpert dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--uncertain-policy", choices=["zero", "ones", "masked"], default="zero")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--output-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = find_dataset_root(args.data_root)

    print(f"Dataset root: {root}")
    print(f"Initializing {args.model} | Device: {DEVICE} | Label policy: {args.uncertain_policy}")

    train_ds = CheXpertDataset(
        csv_path=root / "train.csv",
        data_root=root,
        model_name=args.model,
        max_samples=args.max_train_samples,
        uncertain_policy=args.uncertain_policy,
        train=True,
    )
    valid_ds = CheXpertDataset(
        csv_path=root / "valid.csv",
        data_root=root,
        model_name=args.model,
        max_samples=args.max_valid_samples,
        uncertain_policy=args.uncertain_policy,
        train=False,
    )

    print(f"Training samples: {len(train_ds):,} | Validation samples: {len(valid_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Initialize model
    model = build_model(args.model, pretrained=not args.no_pretrained).to(DEVICE)

    # Learning rate selection
    default_lrs = {"vit": 2e-5, "densenet": 1e-4, "capsnet": 1e-4, "gat": 1e-4}
    lr = args.learning_rate or default_lrs[args.model]

    pos_weight = compute_pos_weight(train_ds).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.model}.pt"

    best_auc = -float("inf")
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_total = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            logits = forward_logits(model, images)
            loss = compute_masked_loss(logits, labels, pos_weight, args.uncertain_policy)
            loss.backward()
            optimizer.step()

            train_total += float(loss.item()) * images.size(0)

        train_loss = train_total / len(train_ds)
        val_loss, summary, per_finding, _, _ = evaluate(
            model=model,
            loader=valid_loader,
            pos_weight=pos_weight,
            device=DEVICE,
            policy=args.uncertain_policy,
        )
        scheduler.step()

        macro_auc = summary.get("macro_auroc")
        current_score = float(macro_auc) if macro_auc is not None else -val_loss
        improved = (current_score > best_auc) if macro_auc is not None else (val_loss < best_loss)

        if improved:
            best_auc = current_score if macro_auc is not None else best_auc
            best_loss = val_loss

            payload = {
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "labels": FINDINGS,
                "epoch": epoch,
                "validation_summary": summary,
                "input_size": 224,
                "uncertain_policy": args.uncertain_policy,
                "timestamp": time.time(),
            }
            torch.save(payload, checkpoint_path)
            per_finding.to_csv(output_dir / f"{args.model}_metrics.csv", index=False)
            (output_dir / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  ✓ Best model saved to {checkpoint_path} (AUROC: {macro_auc})")

        epoch_rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "validation_loss": round(val_loss, 4),
            "validation_macro_auroc": macro_auc,
            "validation_macro_f1": summary.get("macro_f1"),
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": round(time.perf_counter() - started, 1),
            "checkpoint_updated": bool(improved),
        }
        history.append(epoch_rec)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro AUROC: {macro_auc}")

    (output_dir / f"{args.model}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("\nTraining completed.")
    print(f"Best Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
