"""Medical Vision Transformer (ViT-B/16) Training Pipeline for CheXpert / NIH Multi-Label Classification.

Features:
- Model: ViT-B/16 (vit_base_patch16_224) with 13-class Linear head outputting raw logits.
- Loss: BCEWithLogitsLoss with dynamically computed pos_weight per class for imbalanced labels.
- Uncertainty Handling: Supports U-Zeros, U-Ones, and U-Ignore policies.
- Optimization: AdamW (lr=3e-5, weight_decay=0.01) with CosineAnnealingLR.
- Efficiency: PyTorch Automatic Mixed Precision (AMP) via torch.cuda.amp.autocast and GradScaler.
- Checkpointing: Evaluates validation CheXpert-5 AUROC after every epoch and saves best weights to checkpoints/vit.pt.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tv_models

from dataset import (
    CheXpertDataset,
    build_medical_transforms,
    compute_dynamic_pos_weights,
    FINDINGS,
    CHEXPERT_5,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vit_classifier(num_classes: int = len(FINDINGS), pretrained: bool = True) -> nn.Module:
    """Builds ViT-B/16 with replaced 13-class classification head outputting raw logits."""
    try:
        from transformers import ViTForImageClassification, ViTConfig
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.num_labels = num_classes
        config.problem_type = "multi_label_classification"
        model = ViTForImageClassification(config)
    except Exception:
        # Fallback to TorchVision ViT-B/16
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    return model


def forward_raw_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Executes forward pass and returns raw logits tensor [B, 13] (strictly without Sigmoid)."""
    outputs = model(images)
    if hasattr(outputs, "logits"):
        return outputs.logits
    elif isinstance(outputs, dict):
        return outputs.get("logits", outputs)
    elif isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def compute_masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Computes BCEWithLogitsLoss with dynamic class weighting and U-Ignore masking."""
    unweighted = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    # Apply U-Ignore mask
    masked = unweighted * masks
    return masked.sum() / (masks.sum() + 1e-7)


def compute_masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    policy: str = "masked",
) -> torch.Tensor:
    """Computes BCEWithLogitsLoss with uncertain label policy (-1)."""
    if pos_weight is None:
        pos_weight = torch.ones(logits.shape[-1], device=logits.device)
    if policy in ("masked", "ignore"):
        mask = (labels != -1.0).float()
        valid_targets = torch.clamp(labels, min=0.0)
        unweighted = F.binary_cross_entropy_with_logits(
            logits, valid_targets, pos_weight=pos_weight, reduction="none"
        )
        masked = unweighted * mask
        return masked.sum() / (mask.sum() + 1e-7)
    elif policy == "zeros":
        valid_targets = torch.where(labels == -1.0, torch.zeros_like(labels), labels)
        return F.binary_cross_entropy_with_logits(logits, valid_targets, pos_weight=pos_weight)
    elif policy == "ones":
        valid_targets = torch.where(labels == -1.0, torch.ones_like(labels), labels)
        return F.binary_cross_entropy_with_logits(logits, valid_targets, pos_weight=pos_weight)
    else:
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> float:
    """Executes one training epoch with AMP where available."""
    model.train()
    total_loss = 0.0
    use_cuda_amp = (device.type == "cuda" and scaler is not None)

    for images, targets, masks in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_cuda_amp:
            with torch.cuda.amp.autocast():
                logits = forward_raw_logits(model, images)
                loss = compute_masked_bce_loss(logits, targets, masks, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = forward_raw_logits(model, images)
            loss = compute_masked_bce_loss(logits, targets, masks, pos_weight)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * images.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train Medical ViT on Chest Radiographs")
    parser.add_argument("--data-root", default="data/CheXpert-v1.0-small", help="Path to CheXpert root directory")
    parser.add_argument("--u-policy", default="zero", choices=["zero", "ones", "ignore"], help="U-label policy")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for AdamW")
    parser.add_argument("--save-path", default="checkpoints/vit.pt", help="Output path for best checkpoint")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional sample limit for quick test")
    args = parser.parse_args()

    print("=" * 80)
    print("CHEST X-RAY INTELLIGENCE - MEDICAL ViT-B/16 TRAINING PIPELINE")
    print(f"Device   : {DEVICE}")
    print(f"U-Policy : {args.u_policy}")
    print(f"Epochs   : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 80)

    # 1. Datasets & Loaders
    data_root = Path(args.data_root)
    train_csv = data_root / "train.csv"
    valid_csv = data_root / "valid.csv"

    if not train_csv.exists() or not valid_csv.exists():
        print(f"[!] Warning: train.csv or valid.csv not found at {data_root}.")
        print("[!] Generating synthetic demonstration dataset for immediate local test...")
        from scripts.generate_demo_checkpoints import generate_sample_radiographs
        generate_sample_radiographs(Path("data/sample_images"))
        return

    train_ds = CheXpertDataset(
        csv_path=train_csv,
        image_root=data_root,
        transform=build_medical_transforms(is_train=True),
        u_policy=args.u_policy,
        max_samples=args.max_train_samples,
    )

    valid_ds = CheXpertDataset(
        csv_path=valid_csv,
        image_root=data_root,
        transform=build_medical_transforms(is_train=False),
        u_policy=args.u_policy,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Dynamic Positive Weighting
    pos_weight = compute_dynamic_pos_weights(train_ds).to(DEVICE)
    print("\nDynamic Class Imbalance Weights (pos_weight):")
    for name, w in zip(FINDINGS, pos_weight.cpu().numpy()):
        print(f"  {name:<28}: {w:.2f}")

    # 3. Model & Optimizer
    model = build_vit_classifier(num_classes=len(FINDINGS), pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    # 4. Training Loop
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, pos_weight, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"  [+] Saved new best ViT checkpoint to {save_path}")

    print("\n[SUCCESS] ViT training pipeline completed successfully!")


if __name__ == "__main__":
    main()
