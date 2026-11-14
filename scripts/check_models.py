import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api import checkpoint_path
from backend.config import CHECKPOINT_DIR, MODEL_INFO

print(f'Checkpoint folder: {CHECKPOINT_DIR}')
for model_id, info in MODEL_INFO.items():
    path=checkpoint_path(model_id)
    status='READY' if path.is_file() else 'MISSING'
    size=f'{path.stat().st_size/(1024**2):.1f} MB' if path.is_file() else '—'
    print(f'{model_id:10s} | {status:7s} | {info["name"]:42s} | {path.name if path.is_file() else "expected .pt file"} | {size}')
print('Compatibility is checked strictly when the API loads a selected model.')
