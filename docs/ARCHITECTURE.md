# Technical Architecture: Chest X-ray Intelligence Studio

## System Overview
Chest X-ray Intelligence Studio is a modular, research-quality medical AI system combining multi-model deep learning, explainable AI (XAI), and Graph-Based Retrieval-Augmented Generation (GraphRAG).

```text
                                [ Input Radiograph (1–5 JPG/PNG) ]
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Stage 1: Out-of-Distribution (OOD) Filter  │
                         │   - Color Saturation & Chrominance Check     │
                         │   - Thoracic Bimodal Intensity Histogram     │
                         │   - High-Frequency Texture & Edge Entropy    │
                         │   - Cross-Modality Filters (MRI/CT Detector) │
                         └──────────────────────┬───────────────────────┘
                                                │ (Pass)
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Stage 2: Multi-Model Inference Engine      │
                         │   - Vision Transformer (ViT-B/16) [PRIMARY]  │
                         │   - DenseNet121 [CNN Feature Reuse]          │
                         │   - Capsule Network [CapsNet Dynamic Routing]│
                         │   - Graph Attention Network [Image-GAT]      │
                         └──────────────────────┬───────────────────────┘
                                                │ [B, 13] Sigmoid Probabilities
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Stage 3: Explainable AI (XAI) Suite        │
                         │   - Panel 1: Original Clinical Radiograph    │
                         │   - Panel 2: Model-Specific Heatmap          │
                         │   - Panel 3: ViT Attention Rollout (12 Layer)│
                         │   - Panel 4: Fused Diagnostic Visual Overlay │
                         └──────────────────────┬───────────────────────┘
                                                │ Attributions & Findings
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Stage 4: Biomedical Knowledge Graph        │
                         │   - NetworkX Directed Heterogeneous Graph    │
                         │   - 13 Findings linked to Thoracic Anatomy   │
                         │   - Categories & Diagnostic Visual Concepts  │
                         │   - Verified Citations (NCBI & PubMed)       │
                         └──────────────────────┬───────────────────────┘
                                                │ Targeted Subgraph & Passages
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   Stage 5: Evidence-Grounded GraphRAG        │
                         │   - Strict Medical Grounding Guardrails      │
                         │   - Zero Hallucination Deterministic Engine  │
                         │   - Optional Local Instruction LLM (Qwen)    │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                                 [ Structured Clinical Report ]
```

---

## 1. Domain Separation: GAT vs. GraphRAG
A critical architectural principle of this system is the formal distinction between the **Image-Region Graph Attention Network (GAT)** and the **Biomedical Knowledge Graph (GraphRAG)**:

1. **Image-Region GAT (`backend/gat.py`)**:
   - Operates on spatial visual features within a single radiograph.
   - Extracts 49 spatial region nodes (7×7 grid) from convolutional feature maps.
   - Models spatial correlations across the patient's thoracic cavity using multi-head attention.
   - Generates visual-feature representations for disease classification.
2. **Biomedical Knowledge Graph & GraphRAG (`backend/graph_rag.py`)**:
   - Operates on symbolic medical knowledge and clinical literature.
   - Connects semantic disease entities to anatomical concepts, radiographic signs, and NCBI Bookshelf literature sources.
   - Retrieves verified external evidence to ground and explain model predictions.

---

## 2. The 13 CheXpert Finding Classes
The application strictly outputs predictions across the 13 official CheXpert abnormality findings:
1. `Enlarged Cardiomediastinum` (Mediastinum)
2. `Cardiomegaly` (Cardiac silhouette)
3. `Lung Opacity` (Lung parenchyma)
4. `Lung Lesion` (Lung parenchyma)
5. `Edema` (Pulmonary interstitium & alveoli)
6. `Consolidation` (Alveolar airspaces)
7. `Pneumonia` (Lung parenchyma)
8. `Atelectasis` (Lung subsegments)
9. `Pneumothorax` (Pleural space)
10. `Pleural Effusion` (Costophrenic angle / pleural space)
11. `Pleural Other` (Pleural membranes)
12. `Fracture` (Thoracic skeletal cage)
13. `Support Devices` (Thorax - external / invasive hardware)
