# AIML-02: Dataset Documentation & Data Governance Record

**Project Goal:** AI-Assisted Medical Imaging and Diagnosis System  
**Notice:** For research and education only — not for clinical use. Must never be used to diagnose, triage, or guide patient care.

---

## 1. Datasets Overview

The system uses two separate datasets for distinct vision tasks:
1. **Chest Radiograph Multi-Label Classification:** NIH ChestX-ray14
2. **Volumetric Lung Lesion Segmentation:** Medical Segmentation Decathlon (MSD) Task06 Lung CT

These datasets represent different modalities (2D projection radiography vs. 3D helical CT) and different patient cohorts. They are never conflated or merged into a single clinical claim.

---

## 2. NIH ChestX-ray14 Specification

- **Source:** National Institutes of Health (NIH) Clinical Center
- **Access URL:** [NIH ChestX-ray14 Repository](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Archive Version:** 14-disease natural language processed extraction (Wang et al., 2017)
- **Modality:** Frontal Chest Radiographs (DX, CR), Posteroanterior (PA) and Anteroposterior (AP)
- **Resolution:** Grayscale PNG, natively 1024×1024, rescaled to 224×224 for DenseNet-121
- **Target Path:** `data/chestxray/images/` and `data/chestxray/labels.csv`

### Required Schema
| Column | Type | Description | Handling Rule |
| :--- | :--- | :--- | :--- |
| `Image Index` | string | Unique file identifier (e.g., `00000001_000.png`) | Non-null, must exist on disk |
| `Finding Labels` | string | Pipe-delimited list of 14 findings or `No Finding` | One-hot encoded across target vector |
| `Patient ID` | int/str | Anonymized patient cluster identifier | Mandatory grouping key for splits |
| `Patient Age` | int | Patient age in years | Used for demographic stratification |
| `Patient Gender` | string | Biological sex (`M`/`F`) | Audited for demographic parity |
| `View Position` | string | Radiographic projection (`PA`/`AP`) | Filtered/documented during training |

### Class Imbalance & Label Noise
- **Report Mining Noise:** Labels were extracted from radiology reports using NLP (MetaMap/NegBio) without manual radiologist verification for all images, introducing an estimated 10-15% label noise.
- **Extreme Class Imbalance:** Certain conditions (e.g., Hernia: <0.2% prevalence) suffer from severe data sparsity compared to Infiltration (>15%). Models trained on raw frequencies will exhibit high recall on majority classes and false negatives on minority classes.
- **No Finding Dominance:** >50% of the dataset contains `No Finding`, requiring balanced mini-batching or weighted binary cross-entropy loss.

---

## 3. MSD Task06 Lung CT Specification

- **Source:** Medical Segmentation Decathlon (MSD)
- **Access URL:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- **Modality:** 3D Thoracic Computed Tomography (CT) stored in NIfTI format (`.nii.gz`)
- **Target Path:** `data/msd/Task06_Lung/`
- **Annotations:** Voxel-wise binary ground-truth masks for lung lesions.

### Preprocessing Protocol
1. **Reorientation:** RAS orientation using MONAI `Orientationd`.
2. **Resampling:** Uniform voxel spacing of $(1.5, 1.5, 2.0)$ mm using bilinear interpolation for intensity and nearest-neighbor for masks.
3. **Intensity Clipping:** Windowing between $[-1000, 400]$ Hounsfield Units (HU), scaled to $[0.0, 1.0]$.
4. **Spatial Cropping:** Background suppression followed by random balanced positive/negative voxel patches of $(96, 96, 96)$.

---

## 4. Patient-Safe Partitioning & Leakage Prevention

A fundamental flaw in medical machine learning is **patient-identity leakage across train and test partitions**. In ChestX-ray14, individual patients frequently have multiple scans over years.

### Policy Enforcement
- **Split Rule:** All studies belonging to a single `Patient ID` are assigned strictly to train OR validation OR test.
- **Implementation:** `sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` grouping on `Patient ID`.
- **Automated Verification:** Verified by `src/validate_manifest.py::check_split_leakage` and unit tests in `tests/test_manifest.py`.

---

## 5. Ethical Considerations & AI Governance

1. **De-identification:** Only public, open-access, fully anonymized research data is permitted. Direct identifiable health information (PHI) must never be added to repository or logs.
2. **Audit Logging:** Every human review decision is recorded to an append-only JSONL log (`artifacts/audit.jsonl`) with pseudonymized study IDs, timestamp, and review outcome.
3. **Explanations Are Not Proof:** Grad-CAM heatmaps highlight feature attribution in DenseNet convolutional layers. They represent correlation, not histological proof.
