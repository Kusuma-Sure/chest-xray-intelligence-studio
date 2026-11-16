# Chest X-ray Intelligence
## AI-Based Medical Image Classification

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local, high-performance research platform for multi-label chest radiograph classification on the 13 CheXpert findings, featuring dynamic switching across 4 deep learning architectures, 4-panel visual explainability (XAI), Out-of-Distribution (OOD) rejection, and factually grounded Biomedical Knowledge Graph Retrieval-Augmented Generation (GraphRAG).

---

## 1. Project Overview & Key Innovations

- **Application Title**: Chest X-ray Intelligence
- **Domain**: Medical Image Analysis — Frontal Chest Radiography (AP and PA projections)
- **Dataset Integration**: CheXpert-v1.0-small (automated download via KaggleHub)
- **Primary Architecture**: Vision Transformer (ViT-B/16) initialized from Hugging Face `google/vit-base-patch16-224`
- **Baseline & Experimental Architectures**:
  1. DenseNet121 (Dense Convolutional Network feature reuse)
  2. Capsule Network (CapsNet with vector capsules and dynamic routing)
  3. Graph Attention Network (GAT with 49 spatial region nodes and 8-neighbor spatial adjacency)
- **Visual Explainability (4-Panel Grid)**:
  - Panel 1: Original Preprocessed Radiograph
  - Panel 2: Model-Specific Heatmap (Grad-CAM, Grad-CAM++, ViT Grad×Act, GAT Region Attention)
  - Panel 3: ViT Attention Rollout (recursive self-attention across 12 layers) / Secondary Structural Edge Map
  - Panel 4: Fused Diagnostic Visual Overlay (high-contrast alpha blend)
- **Out-of-Distribution (OOD) Guardrail**: Rejection filter identifying non-medical photos (cars, dogs, food, scenery), synthetic diagrams, and unsupported modalities (brain MRI, axial CT).
- **Biomedical Knowledge Graph & GraphRAG**: Directed NetworkX knowledge graph mapping all 13 CheXpert findings to precise thoracic anatomy and verified medical evidence from **NCBI Bookshelf** and **PubMed**, operating under strict non-hallucination clinical guardrails.
- **Multi-Model Benchmark Register**: Empirical held-out validation comparisons (AUROC, AUPRC, F1-score, Precision, Recall).

---

## 2. Project Directory Structure

```text
Chest_Xray_Intelligence_Studio/
│
├── app.py                     # Primary application entry point (Uvicorn / FastAPI)
├── requirements.txt           # Project dependencies
├── benchmark_results.json     # Empirical validation benchmark register
├── README.md                  # Complete execution and research documentation
│
├── backend/
│   ├── architectures.py       # ViT-B/16 (HF), DenseNet121, CapsNet, GAT implementations
│   ├── gat.py                 # Graph Attention Network with 7×7 spatial region nodes
│   ├── config.py              # 13 CheXpert findings, anatomical ontology, model specs
│   ├── data.py                # Dataset loader, patient-level splits, transforms
│   ├── train.py               # Unified training CLI with uncertainty policy support
│   ├── evaluate.py            # Standalone evaluation & per-class metrics reporting
│   ├── predictor.py           # Multi-model inference, batching (1-5 images), OOD integration
│   ├── model_registry.py      # Checkpoint discovery, status tracking, lazy loading & caching
│   ├── explainability.py      # Grad-CAM, Grad-CAM++, ViT Attention Rollout, GAT Attention
│   ├── visualization.py       # 4-panel visual grid generator & colormap overlays
│   ├── ood_detector.py        # Anomaly scoring & Out-of-Distribution image rejection
│   ├── graph_rag.py           # Biomedical Knowledge Graph (NetworkX) & GraphRAG engine
│   ├── benchmark.py           # Metrics aggregation & comparative chart rendering
│   ├── metrics.py             # Validation AUROC, AUPRC, F1, precision, recall formulas
│   └── api.py                 # FastAPI REST routing & endpoints
│
├── frontend/
│   ├── index.html             # High-end dark medical AI dashboard (5 tabs, 1-5 image upload)
│   ├── styles.css             # Dark charcoal, cyan accent clinical theme
│   └── app.js                 # Frontend state controller, model selector, XAI & GraphRAG
│
├── checkpoints/               # Trained model state dictionaries (.pt / .pth)
│   ├── vit_chexpert_baseline/
│   ├── densenet121/
│   ├── capsnet/
│   └── gat/
│
├── docs/
│   ├── SOURCE_AUDIT.md        # Complete codebase audit report & problem matrix
│   ├── ARCHITECTURE.md        # Technical architecture & pipeline schematic
│   ├── TRAINING_GUIDE.md      # Google Colab & Windows local training guide
│   ├── MODEL_DOCUMENTATION.md # Mathematical formulations (Routing, GAT, Rollout)
│   ├── LIMITATIONS.md         # Clinical safety, research disclaimer, edge-case limitations
│   └── medical_evidence.json  # Verified NCBI Bookshelf citations for all 13 findings
│
├── notebooks/
│   └── CheXpert_Colab_Training.ipynb  # Self-healing Colab training notebook
│
├── scripts/
│   ├── check_models.py        # Inspects model registry and checkpoint readiness
│   ├── verify_checkpoints.py  # Tests tensor shapes and checkpoint compatibility
│   ├── download_dataset.py    # Automated KaggleHub dataset fetcher
│   └── run_benchmark.py       # Evaluates checkpoints and generates comparison plots
│
└── tests/
    ├── test_architectures.py  # Verifies all 4 architectures output [B, 13] logits
    ├── test_ood.py            # Verifies rejection of colored photos, blank inputs, & CXR acceptance
    ├── test_explainability.py # Verifies Grad-CAM, Grad-CAM++, Rollout, & 4-panel grid
    ├── test_graph_rag.py      # Verifies Knowledge Graph & GraphRAG factual grounding
    ├── test_inference.py      # Verifies batch inference (1-5 images) & missing weight notices
    ├── test_training.py       # Verifies patient separation, loss masking, & metrics
    └── test_frontend.py       # Verifies FastAPI REST endpoints & HTML serving
```

---

## 3. Windows Laptop Quickstart (PowerShell / CMD)

### Prerequisites
- Windows 10/11 64-bit
- Python 3.10 or 3.11 installed (available via Windows `py` launcher)

### Step 1: Environment Setup
Open PowerShell or Command Prompt inside the project directory:

```powershell
# Create Python virtual environment
py -3.10 -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install project dependencies
python -m pip install -r requirements.txt
```

### Step 2: Verify System Readiness
```powershell
# Check model architecture status
python scripts\check_models.py

# Verify checkpoint compatibility (when weights are present)
python scripts\verify_checkpoints.py

# Run automated test suite
python -m pytest tests\ -v
```

### Step 3: Launch Dashboard
```powershell
python app.py
```

Open your browser to:
- **Studio Dashboard**: <http://127.0.0.1:8000>
- **Interactive API Documentation**: <http://127.0.0.1:8000/docs>
- **Health Endpoint**: <http://127.0.0.1:8000/api/health>

---

## 4. Google Colab GPU Training Workflow

To train the models on GPU without local compute constraints:
1. Open [Google Colab](https://colab.research.google.com).
2. Set runtime type to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Open `notebooks/CheXpert_Colab_Training.ipynb`.
4. Run the notebook. The self-healing script automatically locates `backend/train.py` regardless of folder naming in `/content`.
5. Run the automated training cells for **ViT-B/16**, **DenseNet121**, **CapsNet**, and **GAT**.
6. The notebook zips all trained `.pt` files and automatically triggers a download of `trained_checkpoints.zip`.
7. Extract the checkpoints into the `checkpoints/` folder on your laptop.

---

## 5. Academic Research Disclaimer & Ethical Notice

This software is an academic research prototype developed for a final-year B.Tech CSE (AI & ML) Capstone project. The disease predictions, visual attribution heatmaps, and GraphRAG summaries do NOT constitute medical diagnoses or clinical validation. Results must not be used for patient treatment decisions without comprehensive review by a board-certified radiologist.
