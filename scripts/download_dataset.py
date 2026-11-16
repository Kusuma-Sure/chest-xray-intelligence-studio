"""Downloads CheXpert-v1.0-small dataset using KaggleHub."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import kagglehub
    from backend.data import find_dataset_root
except ImportError:
    print("Please install kagglehub: pip install kagglehub")
    sys.exit(1)

print("Starting CheXpert-v1.0-small automated download via KaggleHub...")
try:
    path = kagglehub.dataset_download("ashery/chexpert")
    print(f"Download completed to: {path}")
    verified_root = find_dataset_root(path)
    print(f"Verified dataset root: {verified_root}")
    print(f"train.csv exists: {(verified_root / 'train.csv').is_file()}")
    print(f"valid.csv exists: {(verified_root / 'valid.csv').is_file()}")
except Exception as exc:
    print(f"Error during dataset download: {exc}")
    print("Ensure Kaggle credentials (kaggle.json) are configured if required by your account.")
