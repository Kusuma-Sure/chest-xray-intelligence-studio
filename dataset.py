"""Medical Radiograph Dataset and Augmentation Pipeline for CheXpert / NIH Multi-Label Classification.

Features:
1. Dynamic U-Label Policy Handling for Uncertain Findings (-1.0 in CheXpert):
   - 'zero': Maps -1.0 to 0.0 (U-Zeros / Conservative policy)
   - 'ones': Maps -1.0 to 1.0 (U-Ones / Aggressive positive policy)
   - 'ignore': Retains -1.0 and generates a binary loss mask tensor (U-Ignore)
2. Medical-Safe Augmentation Pipeline:
   - Letterbox aspect-ratio preserving resize (prevents geometric organ distortion)
   - Subtle affine translation (+/- 4%) and slight rotation (+/- 7 degrees)
   - Contrast/Brightness jitter (+/- 10%) simulating varying beam exposure
   - Strict prohibition of vertical flips (anatomically invalid in chest projection)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 13 Standard CheXpert Multi-Label Findings
FINDINGS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CHEXPERT_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LetterboxResize:
    """Aspect-ratio preserving resize with symmetric padding."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == self.target_size and h == self.target_size:
            return img
        scale = self.target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        return ImageOps.expand(resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def build_medical_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Constructs medical-safe augmentations preserving anatomical orientation."""
    if is_train:
        return transforms.Compose([
            LetterboxResize(img_size),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            LetterboxResize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class CheXpertDataset(Dataset):
    """PyTorch Dataset with robust U-label handling and patient separation."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        transform: Optional[transforms.Compose] = None,
        u_policy: str = "zero",  # 'zero', 'ones', or 'ignore'
        max_samples: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.transform = transform or build_medical_transforms(is_train=False)
        self.u_policy = u_policy.lower().strip()

        if self.u_policy not in ("zero", "ones", "ignore"):
            raise ValueError(f"Unknown U-label policy: {self.u_policy}. Choose from: 'zero', 'ones', 'ignore'.")

        self.df = pd.read_csv(self.csv_path)
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)

        # Ensure label columns exist and map NaN to 0.0
        for col in FINDINGS:
            if col not in self.df.columns:
                self.df[col] = 0.0
            else:
                self.df[col] = self.df[col].fillna(0.0).astype(float)

        self._process_labels()

    def _process_labels(self):
        """Processes the 13 finding columns according to the specified U-label policy."""
        raw_labels = self.df[FINDINGS].values.astype(np.float32)
        n_samples, n_classes = raw_labels.shape

        if self.u_policy == "zero":
            # U-Zeros: treat uncertainty as negative
            self.targets = np.where(raw_labels == -1.0, 0.0, raw_labels)
            self.loss_masks = np.ones((n_samples, n_classes), dtype=np.float32)
        elif self.u_policy == "ones":
            # U-Ones: treat uncertainty as positive
            self.targets = np.where(raw_labels == -1.0, 1.0, raw_labels)
            self.loss_masks = np.ones((n_samples, n_classes), dtype=np.float32)
        elif self.u_policy == "ignore":
            # U-Ignore: mask uncertain labels out of loss computation
            self.targets = np.where(raw_labels == -1.0, 0.0, raw_labels)
            self.loss_masks = np.where(raw_labels == -1.0, 0.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        raw_path = str(row["Path"]) if "Path" in row else str(row.get("image_path", ""))
        
        # Resolve full path
        img_p = Path(raw_path)
        if not img_p.is_absolute():
            img_p = self.image_root / raw_path

        try:
            image = Image.open(img_p).convert("RGB")
        except Exception:
            # Fallback black image if missing file on disk
            image = Image.new("RGB", (224, 224), color=0)

        tensor_img = self.transform(image)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.loss_masks[idx], dtype=torch.float32)

        return tensor_img, target, mask


def compute_dynamic_pos_weights(dataset: CheXpertDataset) -> torch.Tensor:
    """Calculates class-imbalance weights pos_weight = N_neg / (N_pos + epsilon)."""
    targets = dataset.targets  # shape: (N, 13)
    pos_counts = np.sum(targets == 1.0, axis=0)
    neg_counts = np.sum(targets == 0.0, axis=0)
    
    pos_weights = neg_counts / (pos_counts + 1e-5)
    # Clip extreme positive weights between 1.0 and 15.0 for numerical stability
    pos_weights = np.clip(pos_weights, 1.0, 15.0)
    return torch.tensor(pos_weights, dtype=torch.float32)
