"""CheXpert-v1.0-small Dataset Pipeline, Transforms, and Patient Separation.

Supports:
- Automatic KaggleHub dataset path resolution
- Patient-level split protection (prevents image leakage across splits)
- Configurable uncertainty policies (-1 mapped to 0, 1, or masked)
- Architecture-specific transformations (ViT, DenseNet, CapsNet, GAT)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .config import FINDINGS, IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_dataset_root(path: str | Path) -> Path:
    """Discovers the root directory containing train.csv and valid.csv."""
    path = Path(path).expanduser().resolve()
    candidates = [
        path,
        path / "CheXpert-v1.0-small",
        path / "chexpert" / "CheXpert-v1.0-small",
        path / "chexpert",
    ]
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (candidate / "valid.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate train.csv and valid.csv under {path}. "
        "Ensure CheXpert-v1.0-small is downloaded via KaggleHub."
    )


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    """Resolves image paths from CheXpert CSV entries."""
    raw = str(raw_path).replace("\\", "/").strip()
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p
    direct = data_root / Path(raw)
    if direct.is_file():
        return direct
    parts = list(Path(raw).parts)
    lowered = [str(x).lower() for x in parts]
    for idx, part in enumerate(lowered):
        if part == "chexpert-v1.0-small":
            candidate = data_root.joinpath(*parts[idx + 1:])
            if candidate.is_file():
                return candidate
    if len(parts) > 1:
        candidate = data_root.joinpath(*parts[1:])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"CheXpert image path not found: {raw_path} (relative to {data_root})")


def make_transform(model_name: str, train: bool = False):
    """Constructs model-specific torchvision transforms."""
    model_name = model_name.lower().strip()
    if model_name == "capsnet":
        # CapsNet operates on grayscale 1-channel inputs normalized to [0, 1]
        ops = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]
        if train:
            ops += [transforms.RandomRotation(5), transforms.RandomHorizontalFlip(0.2)]
        ops += [transforms.ToTensor()]
    else:
        # ViT, DenseNet, and GAT use 3-channel RGB with standard normalization
        ops = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]
        if train:
            ops += [transforms.RandomRotation(5), transforms.RandomHorizontalFlip(0.2)]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(ops)


def extract_patient_id(path_str: str) -> str:
    """Extracts unique patient identifier (e.g. 'patient12345') to prevent data leakage."""
    match = re.search(r"patient\d+", str(path_str))
    if match:
        return match.group(0)
    return str(path_str)


class CheXpertDataset(Dataset):
    """PyTorch Dataset for CheXpert multi-label radiograph classification."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        model_name: str,
        max_samples: int = 0,
        uncertain_policy: str = "zero",
        train: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.model_name = model_name.lower().strip()
        self.uncertain_policy = uncertain_policy.lower()

        frame = pd.read_csv(csv_path)
        if "Path" not in frame.columns:
            raise ValueError(f"{csv_path} must contain a 'Path' column.")

        missing_labels = [label for label in FINDINGS if label not in frame.columns]
        if missing_labels:
            raise ValueError(f"{csv_path} is missing expected label columns: {missing_labels}")

        # Patient-level subsetting (preserves all images for a selected patient)
        if max_samples and max_samples > 0:
            frame["patient_id"] = frame["Path"].apply(extract_patient_id)
            unique_patients = frame["patient_id"].unique()
            # Sample patients until reaching approximately max_samples
            selected_patients = set()
            count = 0
            for pid in unique_patients:
                selected_patients.add(pid)
                count += (frame["patient_id"] == pid).sum()
                if count >= max_samples:
                    break
            frame = frame[frame["patient_id"].isin(selected_patients)].copy().reset_index(drop=True)

        raw_labels = frame[FINDINGS].apply(pd.to_numeric, errors="coerce")

        if self.uncertain_policy == "ones":
            processed = raw_labels.replace(-1, 1).fillna(0)
        elif self.uncertain_policy == "masked":
            # Keep -1 as a marker for loss masking
            processed = raw_labels.fillna(0)
        else:  # default 'zero'
            processed = raw_labels.replace(-1, 0).fillna(0)

        self.labels = processed.to_numpy(dtype=np.float32)
        self.paths = frame["Path"].astype(str).tolist()
        self.transform = make_transform(self.model_name, train=train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = resolve_image_path(self.paths[index], self.data_root)
        with Image.open(path) as image:
            mode = "L" if self.model_name == "capsnet" else "RGB"
            converted = image.convert(mode)
            tensor = self.transform(converted)
        label_tensor = torch.from_numpy(self.labels[index])
        return tensor, label_tensor


def compute_pos_weight(dataset: CheXpertDataset, max_weight: float = 20.0) -> torch.Tensor:
    """Calculates class balance pos_weight for BCEWithLogitsLoss."""
    valid_labels = dataset.labels.copy()
    valid_labels[valid_labels < 0] = 0  # Treat masked as negative for weight counting
    positives = np.sum(valid_labels == 1, axis=0)
    negatives = np.sum(valid_labels == 0, axis=0)
    ratio = negatives / np.maximum(positives, 1)
    ratio = np.clip(ratio, 1.0, max_weight)
    return torch.tensor(ratio, dtype=torch.float32)
