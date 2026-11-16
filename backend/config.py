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

# Model registry specification with per-model diagnostic calibration
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
        "recommended_threshold": 0.40,
        "description": "Primary architecture. Utilizes multi-head self-attention over 196 image patches (16×16) to capture long-range contextual dependencies across the thoracic cavity.",
        "optimism_bias": {
            "risk": "Global low-frequency contrast inflation across 196 unconstrained patches can over-elevate probabilities on diffuse opacities.",
            "mechanism": "Every patch token directly computes inner-product self-attention with all other patches, causing widespread background or pleural fluid to boost multi-label logits simultaneously.",
            "mitigation": "Calibrated decision threshold of 0.40, patch gradient × activation attribution, and anatomical lung field mask gating to suppress out-of-field artifacts."
        },
        "hidden_dependencies": {
            "clinical_coupling": "Strongly couples bilateral diffuse opacities with secondary findings (e.g., bilateral consolidation and effusion in heart failure).",
            "vulnerability": "Can falsely correlate radio-opaque external hardware (leads, clips) with parenchymal consolidation if attention heads bridge high-density tokens.",
            "mitigation": "Anatomical parenchyma segmentation decouples external dense objects from internal airspace attenuation."
        },
        "unclear_scope": {
            "operational_domain": "Optimal for global multi-focal pattern integration, bilateral interstitial disease, multifocal pneumonia, and subtle hemithorax asymmetry.",
            "out_of_scope": "Not designed for sub-millimeter linear rib fractures or micro-apical pneumothorax visceral lines (<5mm) due to 16×16 patch granularity."
        }
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
        "recommended_threshold": 0.45,
        "description": "Established medical imaging baseline. Connects each layer to every other layer in a feed-forward fashion, ensuring maximum feature reuse and strong gradient propagation.",
        "optimism_bias": {
            "risk": "Feature map concatenation (k=32 growth rate) concentrates gradient magnitudes, creating sharp, overconfident logit peaks.",
            "mechanism": "Repeated concatenation across 4 dense blocks allows high-contrast anatomical edges (rib borders, diaphragmatic domes) to produce extreme activations.",
            "mitigation": "Calibrated decision threshold of 0.45, batch normalization stabilizing channel distributions, and Grad-CAM spatial averaging across denseblock4."
        },
        "hidden_dependencies": {
            "clinical_coupling": "Tightly couples local texture density with disease presence; sensitive to patient rotation and AP vs PA projection geometry.",
            "vulnerability": "Chest rotation projecting mediastinum into the left hemithorax can be misclassified as Cardiomegaly due to local pixel accumulation at the cardiac border.",
            "mitigation": "Anatomical cardiac silhouette segmentation and CTR contour bounding boxes decouple projection shift from true cardiomegaly."
        },
        "unclear_scope": {
            "operational_domain": "Gold standard for localized high-frequency textures: dense lobar consolidations with air bronchograms, focal masses, and sharp costophrenic angle blunting.",
            "out_of_scope": "Suboptimal for cross-hemithorax long-range relational reasoning without deep pooling that erodes fine spatial resolution."
        }
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
        "recommended_threshold": 0.40,
        "description": "Preserves hierarchical part-whole relationships through vector capsules and dynamic routing agreement, preventing spatial information loss common to pooling operations.",
        "optimism_bias": {
            "risk": "Capsule length saturation ||v_j|| -> 1.0 when routing coefficients rapidly converge on dominant structural shapes.",
            "mechanism": "Vector routing by agreement reinforces dominant geometric entities, potentially causing optimism bias on large organs (e.g., cardiac silhouette).",
            "mitigation": "Calibrated threshold of 0.40, margin loss bounds (m+=0.9, m-=0.1), and higher-order Grad-CAM++ with 1st, 2nd, and 3rd order gradient weighting."
        },
        "hidden_dependencies": {
            "clinical_coupling": "Couples disease detection to geometric pose and affine transformation parameters of organ capsules.",
            "vulnerability": "Expects semi-rigid part-whole hierarchies; amorphous, diffuse ground-glass infiltrates lacking clear geometric boundaries can be under-represented.",
            "mitigation": "Pairing capsule outputs with anatomical multi-compartment segmentation ensures unorganized parenchymal opacities are properly tracked."
        },
        "unclear_scope": {
            "operational_domain": "Superb for structural, geometric, and contour-dependent findings: Cardiomegaly, Enlarged Cardiomediastinum, and Pleural Effusion meniscus shapes.",
            "out_of_scope": "Not suitable for non-geometric, diffuse interstitial reticular markings or faint patchy micro-opacities."
        }
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
        "recommended_threshold": 0.38,
        "description": "Transforms image feature maps into a spatial region graph (49 nodes, 8-neighbor edges) and applies multi-head self-attention to model inter-regional thoracic dependencies.",
        "optimism_bias": {
            "risk": "Graph message passing over-smoothing can diffuse focal lesion activations across adjacent lung zones.",
            "mechanism": "Adjacent spatial nodes exchange representations via 8-connectivity edge attention alpha_ij; high activation in one node can artificially elevate neighboring nodes.",
            "mitigation": "Calibrated operating threshold of 0.38, localized multi-head attention entropy normalization, and connected-component boundary extraction."
        },
        "hidden_dependencies": {
            "clinical_coupling": "Directly models anatomical disease spread pathways (e.g., lobar pneumonia spreading to adjacent bronchopulmonary segments or basal effusion layering).",
            "vulnerability": "External linear artifacts (ECG wires, chest tubes) that cross multiple node boundaries can create false edge dependencies between apical and basal nodes.",
            "mitigation": "Anatomical lung field masking prunes non-thoracic peripheral graph nodes prior to regional dependency evaluation."
        },
        "unclear_scope": {
            "operational_domain": "Ideal for topological spread assessment, multi-lobar tracking, and verifying whether an opacity respects anatomical lobar fissures.",
            "out_of_scope": "Spatial resolution is constrained by the 7×7 grid (32×32 pixel nodes); cannot resolve tiny solitary pulmonary nodules (<10mm)."
        }
    },
}

# Configurable uncertainty policies for CheXpert (-1 and NaN handling)
UNCERTAINTY_POLICIES = {
    "zero": "Map uncertain (-1) and missing (NaN) labels to 0 (Conservative / Baseline)",
    "ones": "Map uncertain (-1) to 1 (Sensitive / Aggressive)",
    "masked": "Mask uncertain (-1) labels during loss calculation (True multi-label loss masking)"
}
DEFAULT_UNCERTAINTY_POLICY = "zero"

# ==============================================================================
# EXPANDED THORACIC ONTOLOGY & OPEN-SET VISION-LANGUAGE MAPPINGS
# ==============================================================================
# Additional clinical findings beyond CheXpert 13 (Open-Set Candidates)
ADDITIONAL_THORACIC_FINDINGS = [
    "Emphysema",
    "Pulmonary Fibrosis",
    "Lung Nodule",
    "Lung Mass",
    "Pleural Thickening",
    "Calcification",
    "Hernia",
    "Pneumoperitoneum",
    "Scoliosis",
    "Tuberculosis",
    "Bronchiectasis",
    "Subcutaneous Emphysema",
    "Rib Fracture",
    "Foreign Body",
    "Normal / Unremarkable Findings"
]

ALL_THORACIC_FINDINGS = FINDINGS + ADDITIONAL_THORACIC_FINDINGS

# Extended anatomical and clinical metadata for open-set candidate findings
ADDITIONAL_ANATOMY = {
    "Emphysema": {
        "region": "Lung Parenchyma / Airspaces",
        "subregions": ["Apical Lung Zones", "Centrilobular Spaces", "Panacinar Units"],
        "category": "Chronic Airway Disease",
        "visual_concept": "Hyperlucent, hyperinflated lung fields, flattened diaphragmatic domes, and increased retrosternal airspace",
        "clinical_note": "Abnormal permanent enlargement of airspaces distal to terminal bronchioles with alveolar wall destruction."
    },
    "Pulmonary Fibrosis": {
        "region": "Pulmonary Interstitium",
        "subregions": ["Peripheral Subpleural Basilar Zones", "Bronchovascular Bundles"],
        "category": "Interstitial Lung Disease",
        "visual_concept": "Coarse reticular opacities, traction bronchiectasis, honeycombing, and basilar lung volume loss",
        "clinical_note": "Chronic fibrosing interstitial pneumonia characterized by progressive architectural distortion."
    },
    "Lung Nodule": {
        "region": "Lung Parenchyma",
        "subregions": ["Upper Lobes", "Middle Lobe", "Lower Lobes", "Subpleural Space"],
        "category": "Radiographic Observation",
        "visual_concept": "Well-circumscribed or ill-defined spherical pulmonary opacity measuring less than or equal to 30 mm",
        "clinical_note": "Discrete focal lesion requiring characterization of margins, density, calcification, and comparison with priors."
    },
    "Lung Mass": {
        "region": "Lung Parenchyma / Hilum",
        "subregions": ["Lobar Segments", "Central Hilar Region", "Peripheral Parenchyma"],
        "category": "Radiographic Observation",
        "visual_concept": "Large focal soft-tissue density opacity measuring strictly greater than 30 mm with potential mass effect",
        "clinical_note": "High index of suspicion for primary thoracic neoplasm, large tuberculoma, or fungal abscess; urgent cross-sectional imaging indicated."
    },
    "Pleural Thickening": {
        "region": "Pleural Membranes",
        "subregions": ["Lateral Chest Wall", "Costophrenic Angles", "Apical Caps"],
        "category": "Pleural Pathology",
        "visual_concept": "Continuous or focal band of pleural soft-tissue density along the chest wall (>3mm thickness)",
        "clinical_note": "Chronic fibrocalcific response following hemothorax, chronic infection, or environmental asbestos exposure."
    },
    "Calcification": {
        "region": "Thorax (Parenchymal / Vascular / Nodal)",
        "subregions": ["Granulomatous Foci", "Aortic Arch", "Hilar Lymph Nodes", "Costal Cartilages"],
        "category": "Radiographic Observation",
        "visual_concept": "Dense, well-defined radio-opaque bone-density mineralization within soft tissues or vessel walls",
        "clinical_note": "Healed prior granulomatous infection (e.g., histoplasmosis/TB) or severe vascular atheroma."
    },
    "Hernia": {
        "region": "Diaphragm / Mediastinum",
        "subregions": ["Hiatal Aperture", "Morgagni Foramen", "Bochdalek Foramen"],
        "category": "Anatomical Displacement",
        "visual_concept": "Retrocardiac air-fluid level or soft-tissue mass projecting above the diaphragm contour",
        "clinical_note": "Translocation of abdominal viscera into the thoracic cavity through diaphragmatic defect."
    },
    "Pneumoperitoneum": {
        "region": "Subdiaphragmatic Space",
        "subregions": ["Right Hemidiaphragm Undersurface", "Left Hemidiaphragm Undersurface"],
        "category": "Critical Abdominal Emergency",
        "visual_concept": "Free crescentic gas under the diaphragm domes on erect frontal chest radiograph",
        "clinical_note": "Gastrointestinal perforation until proven otherwise; critical surgical emergency."
    },
    "Scoliosis": {
        "region": "Thoracic Skeletal Cage",
        "subregions": ["Thoracic Spine", "Rib Cage Alignment"],
        "category": "Musculoskeletal Deformity",
        "visual_concept": "Lateral spinal curvature exceeding 10 degrees Cobb angle with rotation of vertebral bodies and rib cage asymmetry",
        "clinical_note": "Structural spinal deviation altering thoracic cavity volumes and cardiopulmonary silhouette position."
    },
    "Tuberculosis": {
        "region": "Lung Parenchyma / Pleura",
        "subregions": ["Apical-Posterior Segments", "Superior Segments Lower Lobes", "Hilar Nodes"],
        "category": "Infectious Disease",
        "visual_concept": "Apical fibro-cavitary consolidation, patchy nodular tree-in-bud opacities, or miliary micronodules",
        "clinical_note": "Mycobacterium tuberculosis infection requiring acid-fast bacilli confirmation and respiratory isolation."
    },
    "Bronchiectasis": {
        "region": "Bronchial Tree",
        "subregions": ["Lower Lobes", "Middle Lobe", "Central Airway Zones"],
        "category": "Airway Pathology",
        "visual_concept": "Tram-track parallel bronchial wall thickening, signet-ring sign, and cystic airspaces with fluid levels",
        "clinical_note": "Irreversible abnormal dilatation of bronchial airways due to recurrent infection or impaired mucociliary clearance."
    },
    "Subcutaneous Emphysema": {
        "region": "Thoracic Soft Tissues",
        "subregions": ["Neck Soft Tissues", "Lateral Chest Wall", "Pectoral Fascia"],
        "category": "Soft Tissue Pathology",
        "visual_concept": "Streaky, dissecting gas lucencies outlining muscle fascicles and soft tissues outside the bony thorax",
        "clinical_note": "Air tracking into subcutaneous planes from pneumothorax, tracheobronchial rupture, or chest wall trauma."
    },
    "Rib Fracture": {
        "region": "Thoracic Skeletal Cage",
        "subregions": ["Posterior Rib Arcs", "Lateral Rib Arcs", "Anterior Rib Arcs"],
        "category": "Traumatic Finding",
        "visual_concept": "Sharp disruption of cortical rib margin with displacement, lucency, or overlapping bone fragment step-off",
        "clinical_note": "Traumatic cortical break requiring inspection for underlying pneumothorax, pulmonary contusion, or flail chest."
    },
    "Foreign Body": {
        "region": "Tracheobronchial Tree / Thorax",
        "subregions": ["Right Mainstem Bronchus", "Left Mainstem Bronchus", "Esophagus", "Chest Wall"],
        "category": "Iatrogenic / Aspirated Object",
        "visual_concept": "Unexpected radio-opaque or non-anatomical object within the airway, gastrointestinal tract, or chest tissues",
        "clinical_note": "Aspirated or ingested exogenous foreign material or surgical retainment requiring urgent retrieval."
    },
    "Normal / Unremarkable Findings": {
        "region": "Entire Thorax",
        "subregions": ["Bilateral Lungs", "Cardiomediastinum", "Pleural Spaces", "Thoracic Cage"],
        "category": "Normal Diagnostic Baseline",
        "visual_concept": "Clear bilateral lung parenchyma, sharp costophrenic angles, normal cardiothoracic ratio (<0.50), intact skeletal margins",
        "clinical_note": "No acute radiographic cardiopulmonary abnormality identified on current projection."
    }
}

COMPLETE_ANATOMY = {**FINDING_ANATOMY, **ADDITIONAL_ANATOMY}

# Standardized clinical radiological prompt templates for vision-language matching
CLINICAL_PROMPT_TEMPLATES = [
    "a frontal chest radiograph showing {finding}",
    "a chest X-ray demonstrating clear signs of {finding}",
    "radiological image of thorax with {finding}",
    "chest radiograph with presence of {finding} in the chest cavity"
]

# Open-Set & Uncertainty Configuration
OPEN_SET_CONFIG = {
    "energy_temperature": 1.0,
    "min_confidence_threshold": 0.35,
    "ood_energy_threshold": -4.5,
    "candidate_similarity_threshold": 0.42,
    "abstention_message": "No reliable prediction — further review required",
    "disclaimer": "AI-generated finding candidate — requires review by a qualified radiologist."
}

