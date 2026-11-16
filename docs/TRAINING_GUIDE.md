# Training Guide: CheXpert Multi-Model Pipeline

This guide documents the procedures for training and evaluating the four architectures (ViT-B/16, DenseNet121, CapsNet, and GAT) on Google Colab GPU or locally on Windows.

---

## 1. Google Colab GPU Workflow (Recommended)

1. Open [Google Colab](https://colab.research.google.com).
2. Set runtime type to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Open or upload `notebooks/CheXpert_Colab_Training.ipynb`.
4. Run Cell 1 and Cell 2 to verify CUDA GPU availability and install dependencies (`kagglehub`, `transformers`, `accelerate`, etc.).
5. If prompted in Cell 3, upload your project ZIP. The self-healing script will automatically locate `backend/train.py` regardless of folder naming.
6. Run Cell 4 to download `CheXpert-v1.0-small` via KaggleHub.
7. Configure training parameters in Cell 5:
   - `SMOKE_TEST = False` for full research runs, or `True` for quick pipeline sanity tests.
   - `UNCERTAIN_POLICY`:
     - `"zero"`: Conservative baseline; maps `-1` and `NaN` to `0`.
     - `"ones"`: High sensitivity; maps `-1` to `1`.
     - `"masked"`: True multi-label masking; skips `-1` labels from BCE loss computation.
8. Train each architecture by running its corresponding cell:
   - **ViT-B/16**: Batch size 16, learning rate `2e-5`, CosineAnnealingLR.
   - **DenseNet121**: Batch size 32, learning rate `1e-4`, CosineAnnealingLR.
   - **CapsNet**: Batch size 16, learning rate `1e-4`, dynamic routing iterations 3.
   - **GAT**: Batch size 16, learning rate `1e-4`, 8-neighbor spatial adjacency.
9. Cell 11 zips all trained `.pt` checkpoints, validation metric CSVs, and summary JSONs into `trained_checkpoints.zip` and triggers an automatic browser download.
10. Unzip `trained_checkpoints.zip` into the `checkpoints/` directory on your local Windows laptop.

---

## 2. Local Training on Windows (CLI)

Ensure your Python environment is activated and CheXpert dataset path is known:

```powershell
# DenseNet121 Training
python -m backend.train --model densenet --data-root "D:\datasets\CheXpert-v1.0-small" --epochs 5 --batch-size 16 --uncertain-policy zero

# Vision Transformer (ViT-B/16) Training
python -m backend.train --model vit --data-root "D:\datasets\CheXpert-v1.0-small" --epochs 5 --batch-size 8 --learning-rate 2e-5

# Capsule Network (CapsNet) Training
python -m backend.train --model capsnet --data-root "D:\datasets\CheXpert-v1.0-small" --epochs 5 --batch-size 16

# Graph Attention Network (GAT) Training
python -m backend.train --model gat --data-root "D:\datasets\CheXpert-v1.0-small" --epochs 5 --batch-size 16
```

---

## 3. Evaluation and Benchmark Refresh

To evaluate an installed checkpoint and output detailed per-class metrics:

```powershell
python -m backend.evaluate --model vit --data-root "D:\datasets\CheXpert-v1.0-small" --threshold 0.50
```

To refresh the comparative benchmark charts displayed on the frontend:

```powershell
python scripts\run_benchmark.py
```
