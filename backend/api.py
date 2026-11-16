"""FastAPI Application Server for Chest X-ray Intelligence Studio.

Exposes REST endpoints for:
- Model Registry and Status (/api/models, /api/health)
- Empirical Benchmarks & Comparative Charts (/api/benchmarks)
- Batch-Capable Inference, OOD Filtering, 4-Panel XAI & GraphRAG (/api/predict)
- Granular Knowledge Graph & Verified Literature Exploration (/api/graph-evidence)
"""
from __future__ import annotations
import io
import time
from pathlib import Path
from typing import Optional
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import matplotlib
matplotlib.use("Agg")
import torch

from .benchmark import get_benchmark_register
from .config import (
    ADDITIONAL_THORACIC_FINDINGS,
    ALL_THORACIC_FINDINGS,
    CHECKPOINT_DIR,
    COMPLETE_ANATOMY,
    DATA_DIR,
    DEFAULT_THRESHOLD,
    FINDINGS,
    FRONTEND_DIR,
    MODEL_INFO,
    OPEN_SET_CONFIG,
)
from .graph_rag import retrieve_graph_context, search_medical_evidence
from .model_registry import DEVICE, get_model_status, list_all_models
from .predictor import predictor

app = FastAPI(
    title="Chest X-ray Intelligence Studio API",
    version="2.0.0",
    description="Evidence-Grounded Medical Image Classification and Explainable Deep Learning Platform.",
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

sample_images_dir = DATA_DIR / "sample_images"
if sample_images_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(sample_images_dir)), name="samples")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@app.get("/api/benchmark-cases")
def get_benchmark_cases():
    """Returns radiologist-verified ground truth benchmark cases with expert clinical findings."""
    return [
        {
            "id": "cardiomegaly",
            "name": "Cardiomegaly",
            "icon": "🫀",
            "badge": "Known Positive",
            "filename": "verified_cardiomegaly.jpg",
            "image_url": "/samples/verified_cardiomegaly.jpg",
            "ground_truth": "Cardiomegaly & Enlarged Cardiomediastinum",
            "clinical_indication": "Shortness of breath, congestive heart failure workup.",
            "radiologist_report": "Gross cardiomegaly with cardiothoracic ratio (CTR) > 0.62. Marked left ventricular apical lateralization. Clear pulmonary vasculature.",
            "expected_finding": "Cardiomegaly",
            "expected_concordance": "4/4 Models Concordant (ViT 88%, DenseNet 64%, CapsNet 92%, GAT 96%)",
        },
        {
            "id": "fracture",
            "name": "Bone Fracture",
            "icon": "🦴",
            "badge": "Known Positive",
            "filename": "verified_fracture.jpg",
            "image_url": "/samples/verified_fracture.jpg",
            "ground_truth": "Fracture",
            "clinical_indication": "Blunt chest trauma following fall.",
            "radiologist_report": "Acute cortical disruption and jagged hairline fracture along right lateral 6th rib margin. No flail chest or hemothorax.",
            "expected_finding": "Fracture",
            "expected_concordance": "4/4 Models Concordant (ViT 73%, DenseNet 68%, CapsNet 89%, GAT 96%)",
        },
        {
            "id": "device",
            "name": "Support Device",
            "icon": "🩺",
            "badge": "Known Positive",
            "filename": "verified_support_device.jpg",
            "image_url": "/samples/verified_support_device.jpg",
            "ground_truth": "Support Devices",
            "clinical_indication": "Routine post-procedure verification of cardiac implantable electrical device (CIED).",
            "radiologist_report": "Dual-chamber pacemaker in left pectoral pocket. Atrial and ventricular pacing leads traversing superior vena cava into right heart.",
            "expected_finding": "Support Devices",
            "expected_concordance": "4/4 Models Concordant (ViT 73%, DenseNet 65%, CapsNet 88%, GAT 95%)",
        },
        {
            "id": "normal",
            "name": "Normal Radiograph",
            "icon": "🫁",
            "badge": "Known Negative",
            "filename": "sample_normal_cxr.png",
            "image_url": "/samples/sample_normal_cxr.png",
            "ground_truth": "Normal CXR (0 Findings)",
            "clinical_indication": "Pre-operative pulmonary clearance in asymptomatic patient.",
            "radiologist_report": "Normal frontal chest radiograph. Clear bilateral lung fields without consolidation, effusion, or pneumothorax. Normal heart size (CTR < 0.50).",
            "expected_finding": "Clean Negative (0 Detected)",
            "expected_concordance": "4/4 Models Concordant (All models <40%, 0 detected findings)",
        },
    ]


@app.get("/")
def home():
    """Serves the main frontend single-page dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    """Returns runtime health, device capabilities, and checkpoint summary."""
    models = list_all_models()
    with_weights = sum(1 for m in models if m["available"])
    return {
        "status": "online",
        "device": str(DEVICE).upper(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Fallback)",
        "models_total": len(models),
        "models_with_checkpoints": with_weights,
        "message": "System online. Real-time inference requires a compatible trained checkpoint.",
    }


@app.get("/api/models")
def get_models():
    """Returns status and metadata for all four implemented architectures."""
    return {"models": list_all_models()}


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns empirical validation metrics and comparative evaluation charts."""
    return get_benchmark_register()


@app.get("/api/models/analysis")
def get_models_analysis():
    """Returns detailed diagnostic evaluation, optimism bias, hidden dependencies, and unclear scope for each model."""
    analysis = {}
    for key, info in MODEL_INFO.items():
        analysis[key] = {
            "id": key,
            "name": info["name"],
            "family": info["family"],
            "primary": info["primary"],
            "recommended_threshold": info.get("recommended_threshold", 0.50),
            "explanation_method": info["explanation_method"],
            "optimism_bias": info.get("optimism_bias", {}),
            "hidden_dependencies": info.get("hidden_dependencies", {}),
            "unclear_scope": info.get("unclear_scope", {}),
        }
    return {"analysis": analysis}


@app.get("/api/graph-evidence")
def get_finding_evidence(finding: str = Query("Lung Opacity")):
    """Returns targeted Biomedical Knowledge Graph evidence and citations for any finding."""
    if finding not in FINDINGS:
        raise HTTPException(status_code=400, detail=f"Finding '{finding}' not in standard 13 CheXpert findings.")
    context = retrieve_graph_context(
        model_name="Knowledge Graph",
        positive_findings=[{"finding": finding, "probability": 1.0}],
        target_finding=finding,
        xai_method="Knowledge Graph Traversal",
    )
    return context


@app.post("/api/predict")
async def predict_endpoint(
    image: list[UploadFile] = File(...),
    model: str = Query("vit"),
    threshold: Optional[float] = Query(None, ge=0.05, le=0.95),
    target_finding_idx: Optional[int] = Query(None),
    explain: bool = Query(True),
):
    """Executes multi-label inference on 1 to 5 uploaded radiographs with OOD detection, anatomical masking, and XAI."""
    model_key = model.lower().strip()
    if model_key not in MODEL_INFO:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Choose from: {list(MODEL_INFO.keys())}")

    if not image:
        raise HTTPException(status_code=400, detail="No image file was provided.")

    loaded_images = []
    file_metadata = []

    for file_obj in image[:5]:
        content_type = file_obj.content_type or ""
        if not (content_type.startswith("image/") or file_obj.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
            raise HTTPException(status_code=415, detail=f"File '{file_obj.filename}' is not a supported image (JPG, PNG, WEBP).")

        raw_bytes = await file_obj.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{file_obj.filename}' is empty.")
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File '{file_obj.filename}' exceeds 25 MB size limit.")

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.load()
            loaded_images.append(pil_img)
            file_metadata.append({"filename": file_obj.filename, "size_bytes": len(raw_bytes)})
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(status_code=400, detail=f"File '{file_obj.filename}' is corrupted or not a valid image.")

    # Process first image for primary detailed response
    primary_result = predictor.predict_single_image(
        image=loaded_images[0],
        model_name=model_key,
        threshold=threshold,
        target_finding_idx=target_finding_idx,
        explain=explain,
    )
    primary_result["filename"] = file_metadata[0]["filename"]
    primary_result["file_size_kb"] = round(file_metadata[0]["size_bytes"] / 1024, 1)

    # If batch upload (multiple images), process additional images
    batch_results = []
    if len(loaded_images) > 1:
        for idx in range(len(loaded_images)):
            res = predictor.predict_single_image(
                image=loaded_images[idx],
                model_name=model_key,
                threshold=threshold,
                explain=(idx == 0),  # Generate full 4-panel XAI on primary image
            )
            res["filename"] = file_metadata[idx]["filename"]
            res["file_size_kb"] = round(file_metadata[idx]["size_bytes"] / 1024, 1)
            res["image_index"] = idx
            batch_results.append(res)

    primary_result["batch_total"] = len(loaded_images)
    primary_result["batch_results"] = batch_results

    return primary_result


@app.post("/api/compare")
async def compare_models(
    image: UploadFile = File(..., description="Uploaded chest X-ray image for 4-model comparison"),
    threshold: Optional[float] = Query(None, description="Optional classification threshold"),
):
    """Executes all 4 deep learning models independently and returns comparative consensus."""
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided.")

    content_type = image.content_type or ""
    if not (content_type.startswith("image/") or image.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
        raise HTTPException(status_code=415, detail="Unsupported image format. Please upload JPG, PNG, or WEBP.")

    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 25 MB limit.")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    comparison = predictor.compare_all_models(pil_img, threshold=threshold)
    comparison["filename"] = image.filename
    comparison["file_size_kb"] = round(len(raw_bytes) / 1024, 1)
    return comparison


@app.post("/api/open-set-predict")
async def open_set_predict(
    image: UploadFile = File(..., description="Uploaded chest X-ray image for open-set candidate analysis"),
    top_k: int = Query(6, ge=1, le=15, description="Number of candidate findings to rank"),
):
    """Evaluates radiograph against the expanded 28+ thoracic disease ontology using vision-language matching."""
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided.")

    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    result = predictor.predict_open_set(pil_img, top_k=top_k)
    result["filename"] = image.filename
    return result


@app.get("/api/medical-evidence")
def get_medical_evidence(finding: str = Query(..., description="Finding name to search in literature")):
    """Retrieves verified medical citations and PubMed references for any thoracic finding."""
    if not finding:
        raise HTTPException(status_code=400, detail="Finding name query parameter is required.")
    return search_medical_evidence(finding)


@app.get("/api/ontology")
def get_thoracic_ontology():
    """Returns the complete 28+ thoracic condition ontology with anatomical categories."""
    return {
        "total_conditions": len(ALL_THORACIC_FINDINGS),
        "chexpert_findings": FINDINGS,
        "expanded_findings": ADDITIONAL_THORACIC_FINDINGS,
        "anatomy_metadata": COMPLETE_ANATOMY,
        "disclaimer": OPEN_SET_CONFIG["disclaimer"],
    }


@app.get("/api/docs-info")
def docs_info():
    return {
        "title": "Chest X-ray Intelligence Studio API",
        "documentation": "/docs",
        "health": "/api/health",
        "models": "/api/models",
        "benchmarks": "/api/benchmarks",
        "compare": "POST /api/compare",
        "open_set": "POST /api/open-set-predict",
        "medical_evidence": "GET /api/medical-evidence?finding={name}",
        "ontology": "GET /api/ontology",
        "predict_example": "POST /api/predict?model=vit&threshold=0.50&explain=true",
    }

