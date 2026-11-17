"""Convenience entry point. Starts the Chest X-ray Intelligence Studio web application."""
import argparse
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for multi-threaded FastAPI
import uvicorn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start Chest X-ray Intelligence Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port number")
    args = parser.parse_args()

    print(f"================================================================")
    print(f"CHEST X-RAY INTELLIGENCE")
    print(f"Opening on: http://{args.host}:{args.port}")
    print(f"API Docs  : http://{args.host}:{args.port}/docs")
    print(f"================================================================")

    # Automatically ensure baseline checkpoints and samples exist so models are immediately available
    try:
        from pathlib import Path
        from backend.config import CHECKPOINT_DIR, DATA_DIR
        needed = ["vit.pt", "densenet.pt", "capsnet.pt", "gat.pt"]
        if any(not (CHECKPOINT_DIR / f).exists() for f in needed):
            print("[+] Initializing baseline deep learning model weights...")
            from scripts.generate_demo_checkpoints import generate_checkpoints_for_dir, generate_sample_radiographs
            generate_checkpoints_for_dir(CHECKPOINT_DIR)
            generate_sample_radiographs(DATA_DIR / "sample_images")
    except Exception as exc:
        print(f"[!] Baseline model initialization notice: {exc}")

    uvicorn.run('backend.api:app', host=args.host, port=args.port, reload=False)
