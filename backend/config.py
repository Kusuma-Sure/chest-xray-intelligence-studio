from __future__ import annotations
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"

IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.50

# 13 CheXpert Multi-Label Findings (strict order)
FINDINGS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Granular Biomedical Knowledge Graph ontology
# Maps each finding to specific anatomical structures, categories, and visual concepts
FINDING_ANATOMY = {
    "Enlarged Cardiomediastinum": {
        "region": "Mediastinum",
        "subregions": ["Superior Mediastinum", "Anterior Mediastinum", "Middle Mediastinum", "Posterior Mediastinum"],
        "category": "Radiographic Observation",
        "visual_concept": "Mediastinal widening (>8cm on AP radiograph or >6cm on PA radiograph)",
        "clinical_note": "Widened contour of thoracic mediastinum requiring distinction from vascular ectasia, lymphadenopathy, or projection artifact."
    },
    "Cardiomegaly": {
        "region": "Cardiac Silhouette",
        "subregions": ["Left Ventricle", "Right Ventricle", "Pericardium", "Left Atrium"],
        "category": "Radiographic Observation",
        "visual_concept": "Cardiothoracic ratio (CTR) exceeding 0.50 on PA projection or 0.55-0.60 on AP projection",
        "clinical_note": "Transverse cardiac diameter greater than half the internal thoracic diameter at the diaphragm level."
    },
    "Lung Opacity": {
        "region": "Lung Parenchyma",
        "subregions": ["Upper Lobe", "Middle Lobe / Lingula", "Lower Lobe", "Alveolar Space", "Interstitium"],
        "category": "Radiographic Observation",
        "visual_concept": "Increased pulmonary attenuation obscuring underlying bronchovascular markings",
        "clinical_note": "Non-specific radiographic density representing airspace filling, interstitial thickening, or atelectasis."
    },
    "Lung Lesion": {
        "region": "Lung Parenchyma",
        "subregions": ["Pulmonary Interstitium", "Bronchial Tree", "Peripheral Lung"],
        "category": "Radiographic Observation",
        "visual_concept": "Discrete solitary or multiple circumscribed opacity (nodule <3cm or mass >3cm)",
        "clinical_note": "Focal mass or nodular density requiring evaluation for margins, calcification, and comparison with priors."
    },
    "Edema": {
        "region": "Pulmonary Interstitium and Alveoli",
        "subregions": ["Perihilar Region", "Interlobular Septa", "Bronchovascular Bundles", "Lower Lung Zones"],
        "category": "Radiographic Observation",
        "visual_concept": "Peribronchial cuffing, Kerley B lines, bat-wing perihilar haze, and vascular redistribution",
        "clinical_note": "Fluid accumulation in pulmonary interstitial tissues progressing to alveolar flooding."
    },
    "Consolidation": {
        "region": "Alveolar Airspaces",
        "subregions": ["Lobar Airspaces", "Segmental Airspaces", "Acinar Units"],
        "category": "Radiographic Observation",
        "visual_concept": "Homogeneous dense airspace filling with visible air bronchograms and silhouette sign",
        "clinical_note": "Alveolar air replaced by fluid, pus, blood, or cells, resulting in tissue-density appearance."
    },
    "Pneumonia": {
        "region": "Lung Parenchyma",
        "subregions": ["Bronchopulmonary Segments", "Lobar Parenchyma", "Subpleural Space"],
        "category": "Disease",
        "visual_concept": "Patchy or confluent focal alveolar infiltrate with ill-defined borders and air bronchograms",
        "clinical_note": "Infectious parenchymal inflammation diagnosed by correlating radiographic opacity with clinical infection signs."
    },
    "Atelectasis": {
        "region": "Lung Parenchyma",
        "subregions": ["Subsegmental Plates", "Lobar Segments", "Basilar Zones"],
        "category": "Radiographic Observation",
        "visual_concept": "Linear horizontal bands (plate-like) or triangular wedge of volume loss with mediastinal/fissural shift",
        "clinical_note": "Partial or complete collapse of lung tissue causing loss of volume and localized density increase."
    },
    "Pneumothorax": {
        "region": "Pleural Space",
        "subregions": ["Apical Pleural Space", "Lateral Thoracic Wall", "Costophrenic Angle"],
        "category": "Radiographic Observation",
        "visual_concept": "Sharp visceral pleural white line with absent peripheral lung markings and hyperlucent space",
        "clinical_note": "Abnormal accumulation of air in the pleural space between the visceral and parietal pleura."
    },
    "Pleural Effusion": {
        "region": "Pleural Space",
        "subregions": ["Costophrenic Sulcus", "Posterior Hemithorax", "Subpulmonic Space"],
        "category": "Radiographic Observation",
        "visual_concept": "Blunting of the lateral or posterior costophrenic angle and upward-curving meniscus sign",
        "clinical_note": "Pathologic fluid accumulation exceeding normal pleural lubricant (>50ml for blunting on erect radiograph)."
    },
    "Pleural Other": {
        "region": "Pleural Membranes",
        "subregions": ["Parietal Pleura", "Visceral Pleura", "Diaphragmatic Pleura"],
        "category": "Radiographic Observation",
        "visual_concept": "Non-effusion pleural pathology including focal plaque calcification, thickening, or fibrothorax",
        "clinical_note": "Fibrotic or calcified pleural abnormalities, often secondary to prior hemothorax, empyema, or asbestos exposure."
    },
    "Fracture": {
        "region": "Thoracic Skeletal Cage",
        "subregions": ["Ribs", "Clavicles", "Thoracic Spine", "Scapula", "Sternum"],
        "category": "Radiographic Observation",
        "visual_concept": "Disruption of cortical bone margin, lucent fracture line, or cortical step deformity",
        "clinical_note": "Bony trauma or insufficiency disruption requiring inspection of rib contours, clavicular symmetry, and spine alignment."
    },
    "Support Devices": {
        "region": "Thorax (External / Invasive Devices)",
        "subregions": ["Trachea", "Superior Vena Cava", "Esophagus", "Pleural Cavity", "Precordium"],
        "category": "Medical Device Finding",
        "visual_concept": "Radio-opaque lines, tips, leads, or hardware (ETT, CVC, NG tube, pacemaker, chest drainage tube)",
        "clinical_note": "Iatrogenic or monitoring hardware verified for anatomic tip position, trajectory, and absence of complications."
    }
}

# Model registry specification
MODEL_INFO = {
    "vit": {
        "id": "vit",
        "name": "Vision Transformer (ViT-B/16)",
        "family": "Vision Transformer",
        "primary": True,
        "checkpoint": "vit.pt",
        "hf_model_id": "google/vit-base-patch16-224",
        "input_type": "RGB / HF ViT Normalization",
        "input_size": 224,
        "explanation_method": "ViT Attention Rollout & Gradient × Activation",
        "description": "Primary architecture. Utilizes multi-head self-attention over 196 image patches (16×16) to capture long-range contextual dependencies across the thoracic cavity.",
    },
    "densenet": {
        "id": "densenet",
        "name": "DenseNet121",
        "family": "Densely Connected CNN",
        "primary": False,
        "checkpoint": "densenet.pt",
        "alternate_checkpoints": ["best_densenet121_chexpert.pth"],
        "input_type": "RGB / ImageNet Normalization",
        "input_size": 224,
        "explanation_method": "Grad-CAM (denseblock4)",
        "description": "Established medical imaging baseline. Connects each layer to every other layer in a feed-forward fashion, ensuring maximum feature reuse and strong gradient propagation.",
    },
    "capsnet": {
        "id": "capsnet",
        "name": "Capsule Network (CapsNet)",
        "family": "Capsule Network with Dynamic Routing",
        "primary": False,
        "checkpoint": "capsnet.pt",
        "alternate_checkpoints": ["best_capsnet_10epoch.pth", "best_capsnet_weighted.pth", "best_capsnet_chexpert.pth"],
        "input_type": "Grayscale / Normalized [0, 1]",
        "input_size": 224,
        "explanation_method": "Grad-CAM++ (Conv3)",
        "description": "Preserves hierarchical part-whole relationships through vector capsules and dynamic routing agreement, preventing spatial information loss common to pooling operations.",
    },
    "gat": {
        "id": "gat",
        "name": "Graph Attention Network (GAT)",
        "family": "Graph Neural Network + CNN",
        "primary": False,
        "checkpoint": "gat.pt",
        "alternate_checkpoints": ["best_gat_chexpert.pth"],
        "input_type": "RGB / ImageNet Normalization",
        "input_size": 224,
        "explanation_method": "GAT Regional Attention Weights & Spatial Node Mapping",
        "description": "Transforms image feature maps into a spatial region graph (49 nodes, 8-neighbor edges) and applies multi-head self-attention to model inter-regional thoracic dependencies.",
    },
}

# Configurable uncertainty policies for CheXpert (-1 and NaN handling)
UNCERTAINTY_POLICIES = {
    "zero": "Map uncertain (-1) and missing (NaN) labels to 0 (Conservative / Baseline)",
    "ones": "Map uncertain (-1) to 1 (Sensitive / Aggressive)",
    "masked": "Mask uncertain (-1) labels during loss calculation (True multi-label loss masking)"
}
DEFAULT_UNCERTAINTY_POLICY = "zero"
