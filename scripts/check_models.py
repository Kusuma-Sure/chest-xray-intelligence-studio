"""Script to check available model architectures and checkpoint readiness."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHECKPOINT_DIR
from backend.model_registry import list_all_models

print("=" * 80)
print("CHEST X-RAY INTELLIGENCE STUDIO -- MODEL REGISTRY AUDIT")
print("=" * 80)
print(f"Checkpoints Directory: {CHECKPOINT_DIR}\n")

models = list_all_models()
for m in models:
    status_icon = "[OK]" if m["available"] else "[--]"
    primary_tag = " [PRIMARY]" if m["primary"] else ""
    print(f"{status_icon} {m['name']}{primary_tag}")
    print(f"    Architecture ID : {m['id']}")
    print(f"    Family          : {m['family']}")
    print(f"    Status          : {m['status']}")
    print(f"    Expected File   : {m['checkpoint_file']}")
    print(f"    XAI Method      : {m['explanation_method']}")
    print(f"    Details         : {m['reason']}")
    print("-" * 80)

available_count = sum(1 for m in models if m["available"])
print(f"\nSummary: {available_count} of {len(models)} model checkpoints installed.")
if available_count == 0:
    print("NOTE: Application starts in diagnostic mode. Train checkpoints using notebooks/CheXpert_Colab_Training.ipynb.")
