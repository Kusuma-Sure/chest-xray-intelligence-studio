"""Runs evaluation across installed checkpoints and exports benchmark_results.json."""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.benchmark import get_benchmark_register, load_benchmark_database

print("=" * 80)
print("BENCHMARK AGGREGATION & REPORT")
print("=" * 80)

register = get_benchmark_register()
print(f"Provenance: {register.get('provenance')}\n")

print(f"{'Model Name':35s} | {'AUROC':7s} | {'AUPRC':7s} | {'Macro F1':8s} | {'Micro F1':8s}")
print("-" * 80)
for m in register.get("models", []):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    print(f"{m['name']:35s} | {fmt(m.get('macro_auroc')):7s} | {fmt(m.get('macro_auprc')):7s} | {fmt(m.get('macro_f1')):8s} | {fmt(m.get('micro_f1')):8s}")

print("\nBenchmark register loaded successfully.")
