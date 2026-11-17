"""Verifies compatibility, keys, and tensor shapes of installed model checkpoints."""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHECKPOINT_DIR, FINDINGS
from backend.model_registry import resolve_checkpoint_path, load_trained_model

print("=" * 80)
print("CHECKPOINT VERIFICATION & COMPATIBILITY CHECK")
print("=" * 80)

models = ["vit", "densenet", "capsnet", "gat"]
for model_id in models:
    path = resolve_checkpoint_path(model_id)
    print(f"\nVerifying '{model_id}'...")
    if path is None or not path.exists():
        print(f"  [MISSING] No checkpoint found at {path}")
        continue

    print(f"  [FOUND] Checking weights at {path} ({path.stat().st_size / (1024*1024):.1f} MB)...")
    try:
        model = load_trained_model(model_id)
        # Test synthetic forward pass
        test_in = torch.zeros(1, 1 if model_id == "capsnet" else 3, 224, 224)
        with torch.no_grad():
            out = model(test_in)
            logits = out["logits"] if isinstance(out, dict) else out
            if isinstance(logits, tuple):
                logits = logits[0]
        if logits.shape == (1, len(FINDINGS)):
            print(f"  [PASSED] Successfully loaded and verified [1, 13] logits output.")
        else:
            print(f"  [WARNING] Output shape was {logits.shape}, expected (1, 13).")
    except Exception as exc:
        print(f"  [ERROR] loading checkpoint: {exc}")

print("\nVerification check complete.")
