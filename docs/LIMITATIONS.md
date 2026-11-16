# Academic Limitations, Safety Guardrails, and Ethical Governance

## 1. Research Prototype Scope
The **Chest X-ray Intelligence Studio** is developed as an academic and educational capstone research prototype in Computer Science & Engineering (AI & ML). 

> [!WARNING]
> This application is NOT an FDA-cleared, CE-marked, or clinically validated medical device. It must NEVER be used as a standalone diagnostic tool, for clinical triage, or for guiding patient treatment.

---

## 2. Model and Attribution Limitations

1. **Attribution Maps $\neq$ Pathology Boundaries**:
   - Visual attribution heatmaps (Grad-CAM, Grad-CAM++, ViT Attention Rollout, GAT regional attention) indicate the computational sensitivity of intermediate neural layers to specific output logits.
   - High attribution scores indicate feature reliance, NOT histological proof or radiologist-confirmed lesion contours.
2. **Model Probabilities $\neq$ Diagnostic Certainty**:
   - The reported probability scores represent sigmoid outputs from models trained on CheXpert-v1.0-small.
   - They reflect conditional likelihood under the training distribution and label policy, not patient prognostic certainty.
3. **CheXpert Dataset Specificity**:
   - CheXpert consists of frontal and lateral chest radiographs from hospital patients.
   - Performance may degrade on pediatric radiographs, portable emergency films with high respiratory motion artifact, or non-hospital screening settings.
4. **Modality Constraint**:
   - This prototype exclusively supports frontal chest radiographs (AP and PA projections).
   - Computed Tomography (CT), Magnetic Resonance Imaging (MRI), Positron Emission Tomography (PET), and Ultrasound are NOT supported.

---

## 3. Strict GraphRAG Guardrails

To prevent hallucination in medical explanations, the GraphRAG pipeline enforces the following immutable guardrails:
1. **Zero Hallucination Guarantee**: Explanations synthesize only facts retrieved directly from the Biomedical Knowledge Graph and peer-reviewed NCBI Bookshelf / PubMed records.
2. **Missing Evidence Protocol**: If external literature or anatomical mapping is absent for a user query or edge-case finding, the engine explicitly outputs:
   > *"The supplied evidence does not establish this."*
3. **No Unwarranted Inferences**: Radiographic findings (such as Lung Opacity) are never conflated with definitive clinical diagnoses (such as Bacterial Pneumonia or Pulmonary Embolism) unless supported by verified external citations and clinical confirmation.
4. **Device Integrity**: The engine never asserts the presence of invasive medical devices (endotracheal tubes, catheters, pacemakers) unless `Support Devices` is both detected above threshold and visually supported.

---

## 4. Patient Privacy and Data Security
- All inference runs locally on the host machine.
- No user-uploaded radiograph images or identifiable patient metadata are transmitted to external cloud endpoints or proprietary third-party LLMs.
- Temporary files and memory tensors are cleared following prediction generation.
