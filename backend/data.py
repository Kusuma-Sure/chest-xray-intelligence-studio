from __future__ import annotations
from pathlib import Path
from typing import Optional
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
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "CheXpert-v1.0-small", path / "chexpert" / "CheXpert-v1.0-small"]
    for candidate in candidates:
        if (candidate / "train.csv").is_file() and (candidate / "valid.csv").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find train.csv and valid.csv under {path}. Expected a CheXpert-v1.0-small directory.")


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    raw = str(raw_path).replace("\\", "/").strip()
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p
    direct = data_root / Path(raw)
    if direct.is_file():
        return direct
    parts = list(Path(raw).parts)
    # Handle CSV paths that include a leading CheXpert-v1.0-small folder.
    lowered = [str(x).lower() for x in parts]
    for idx, part in enumerate(lowered):
        if part == "chexpert-v1.0-small":
            candidate = data_root.joinpath(*parts[idx + 1:])
            if candidate.is_file():
                return candidate
    # Common KaggleHub extraction pattern: <root>/CheXpert-v1.0-small/<Path>
    if len(parts) > 1:
        candidate = data_root.joinpath(*parts[1:])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image path from CSV not found: {raw_path}. Resolved relative to {data_root}.")


def make_transform(model_name: str, train: bool = False):
    if model_name == "capsnet":
        ops = [transforms.Grayscale(num_output_channels=1), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops += [transforms.RandomRotation(5)]
        ops += [transforms.ToTensor()]
    else:
        ops = [transforms.Grayscale(num_output_channels=3), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if train:
            ops += [transforms.RandomRotation(5)]
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class CheXpertDataset(Dataset):
    def __init__(self, csv_path: str | Path, data_root: str | Path, model_name: str,
                 max_samples: int = 0, train: bool = False) -> None:
        self.data_root = Path(data_root).resolve()
        frame = pd.read_csv(csv_path)
        if "Path" not in frame.columns:
            raise ValueError(f"{csv_path} must contain a 'Path' column.")
        missing = [label for label in FINDINGS if label not in frame.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing label columns: {missing}")
        if max_samples and max_samples > 0:
            frame = frame.iloc[:max_samples].copy()
        labels = frame[FINDINGS].apply(pd.to_numeric, errors="coerce").fillna(0).replace(-1, 0)
        self.labels = labels.to_numpy(dtype=np.float32)
        self.paths = frame["Path"].astype(str).tolist()
        self.transform = make_transform(model_name, train=train)
        self.model_name = model_name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = resolve_image_path(self.paths[index], self.data_root)
        with Image.open(path) as image:
            image = image.convert("L" if self.model_name == "capsnet" else "RGB")
            tensor = self.transform(image)
        label = torch.from_numpy(self.labels[index])
        return tensor, label


def compute_pos_weight(dataset: CheXpertDataset, max_weight: float = 20.0) -> torch.Tensor:
    positives = np.sum(dataset.labels == 1, axis=0)
    negatives = np.sum(dataset.labels == 0, axis=0)
    ratio = negatives / np.maximum(positives, 1)
    ratio = np.clip(ratio, 1.0, max_weight)
    return torch.tensor(ratio, dtype=torch.float32)
