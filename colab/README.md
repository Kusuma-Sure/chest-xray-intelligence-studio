# Google Colab Execution Suites: Chest X-ray Intelligence

This directory provides Google Colab notebooks and standalone Python model scripts to demonstrate and evaluate **Chest X-ray Intelligence**:

---

## 1. Notebook Options for Google Colab

### Option A: Full Backend Models & Architecture Notebook (Recommended for Faculty Review)
- **Notebook File**: [`Chest_Xray_Intelligence_Full_Backend_Models.ipynb`](./Chest_Xray_Intelligence_Full_Backend_Models.ipynb)
- **Badge Link**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VEDANTH0517/chest-xray-intelligence-studio/blob/main/colab/Chest_Xray_Intelligence_Full_Backend_Models.ipynb)
- **Description**: Contains the **complete, self-contained Python source code** of all 4 deep learning models (ViT-B/16, DenseNet121, CapsNet with dynamic routing, and GAT with 49-node spatial region graphs), medical image preprocessing (PSNR ≥ 40 dB), OOD guardrail, open-set discovery with Helmholtz free energy, anatomical segmentation with labeled bounding boxes, and 4-panel visual XAI heatmaps. Runs top-to-bottom directly in Colab without needing local files.

### Option B: Live GPU Backend Server & Cloudflare Public HTTPS Tunnel
- **Notebook File**: [`Run_Backend_Google_Colab.ipynb`](./Run_Backend_Google_Colab.ipynb)
- **Badge Link**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VEDANTH0517/chest-xray-intelligence-studio/blob/main/colab/Run_Backend_Google_Colab.ipynb)
- **Description**: Runs the live FastAPI server on an NVIDIA GPU (T4 / V100 / A100) and exposes the full interactive UI via an instant, secure public HTTPS URL.

---

## 2. Separated Standalone Model Python Scripts (`standalone_models/`)

For direct code inspection, modular imports, or running individual architectures separately:
- [`vit_b16.py`](./standalone_models/vit_b16.py): Vision Transformer (ViT-B/16) PyTorch module.
- [`densenet121.py`](./standalone_models/densenet121.py): DenseNet121 convolutional feature reuse module.
- [`capsnet.py`](./standalone_models/capsnet.py): Capsule Network with 784 primary capsules, 16 disease capsules, and dynamic routing.
- [`gat.py`](./standalone_models/gat.py): Graph Attention Network (GAT) with 49-node spatial region graph and 8-neighbor attention.
- [`model_factory.py`](./standalone_models/model_factory.py): Unified factory function `build_model(model_name)` to instantiate any architecture.

