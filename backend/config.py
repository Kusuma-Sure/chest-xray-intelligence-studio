from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.50

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

MODEL_INFO = {
    "vit": {"name": "Vision Transformer (ViT-B/16)", "family": "Vision Transformer", "checkpoint": "vit.pt", "input": "RGB / ImageNet normalized"},
    "densenet": {"name": "DenseNet121", "family": "Densely Connected CNN", "checkpoint": "densenet.pt", "input": "RGB / ImageNet normalized"},
    "capsnet": {"name": "CapsNet", "family": "Capsule Network", "checkpoint": "capsnet.pt", "input": "Grayscale / [0, 1]"},
    "gat": {"name": "Graph Attention Network (GAT)", "family": "Graph Neural Network + CNN", "checkpoint": "gat.pt", "input": "RGB / ImageNet normalized"},
}

# A coarse display taxonomy used only to connect labels to graph nodes. It is not
# a localization claim and is not clinical evidence.
FINDING_REGIONS = {
    "Enlarged Cardiomediastinum": "Cardiomediastinal region",
    "Cardiomegaly": "Cardiac silhouette",
    "Lung Opacity": "Lung fields",
    "Lung Lesion": "Lung fields",
    "Edema": "Lung fields",
    "Consolidation": "Lung fields",
    "Pneumonia": "Lung fields",
    "Atelectasis": "Lung fields",
    "Pneumothorax": "Pleural space",
    "Pleural Effusion": "Pleural space",
    "Pleural Other": "Pleura",
    "Fracture": "Osseous structures",
    "Support Devices": "Support devices",
}
