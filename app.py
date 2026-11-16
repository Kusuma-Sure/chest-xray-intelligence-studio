"""Convenience entry point. Starts the Chest X-ray Intelligence Studio web application."""
import argparse
import os
import uvicorn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start Chest X-ray Intelligence Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port number")
    args = parser.parse_args()

    print(f"================================================================")
    print(f"CHEST X-RAY INTELLIGENCE STUDIO")
    print(f"Opening on: http://{args.host}:{args.port}")
    print(f"API Docs  : http://{args.host}:{args.port}/docs")
    print(f"================================================================")
    uvicorn.run('backend.api:app', host=args.host, port=args.port, reload=False)
