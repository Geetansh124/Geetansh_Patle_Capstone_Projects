# AIML-02 — AI-Assisted Medical Imaging and Diagnosis System

**Research & Educational Prototype**  
**Lead Author:** Geetansh Patle  
**Track:** AI / Machine Learning Capstone  
**Status:** Runnable baseline & verification suite ready (Models require Kaggle/GPU compute for full-scale NIH/MSD checkpoints).

> **CRITICAL SAFETY NOTICE:**  
> **For research and education only — not for clinical use.**  
> This software is an educational prototype and capstone project. It must **never** be used to diagnose, triage, or guide clinical patient care. All confidence scores represent model correlation, not medical certainty.

---

## 1. System Architecture

```text
[ DICOM Studies / Orthanc PACS Gateway ] OR [ Direct Image Upload ]
                      |
                      v
          [ Study Metadata Validation ]
             (Modality DX/CR/CT check)
                      |
         +------------+------------+
         |                         |
         v                         v
 [ X-ray Classification ]   [ 3D Lung CT Segmentation ]
   DenseNet-121 Backbone        MONAI 3D U-Net (RAS 1.5mm)
         |                         |
         v                         v
 [ Grad-CAM Heatmap ]       [ Volumetric Mask Overlay ]
         |                         |
         +------------+------------+
                      |
                      v
          [ FastAPI Inference Gateway ]
                      |
       +--------------+--------------+
       |                             |
       v                             v
[ React Educational Viewer ]  [ Append-Only Audit Trail ]
 (Dual-modal, Opacity slider)   (artifacts/audit.jsonl)
```

---

## 2. Directory Layout

- `code/src/api.py`: FastAPI service with DICOM validation, classification, Grad-CAM, segmentation, and review logging.
- `code/src/train_classifier.py`: DenseNet-121 multi-label classifier training with patient grouping.
- `code/src/evaluate_classifier.py`: Evaluates AUROC, precision, recall, F1, and confusion matrices.
- `code/src/gradcam.py`: Layer-wise Grad-CAM gradient overlay generator.
- `code/src/segmentation_baseline.py`: MONAI 3D U-Net pipeline for MSD Task06 Lung CT.
- `code/src/validate_manifest.py`: Manifest validator, label distribution counter, and split leakage detector.
- `code/frontend/`: Modern React educational viewer with opacity slider and audit history.
- `code/tests/`: Automated unit and integration test suite (`pytest`).
- `data_documentation/DATA.md`: Detailed schema, provenance, split policy, and ethical governance rules.

---

## 3. Quickstart & Verification

### Local Setup
```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Run automated test suite
pytest code/tests

# 3. Generate synthetic sample radiographs and checkpoint
python code/scripts/generate_synthetic_fixtures.py

# 4. Validate synthetic manifest and detect leakage
python code/src/validate_manifest.py --csv data/synthetic_chestxray/labels.csv
```

### Running Services
```bash
# Start Orthanc DICOM server and FastAPI backend
docker compose -f code/docker-compose.yml up --build

# Or start the API directly:
uvicorn code.src.api:app --host 0.0.0.0 --port 8001
```

---

## 4. Truthful Status & Deliverable Checklist

| Deliverable | Required Spec | Current Status |
| :--- | :--- | :--- |
| **Code Pipeline** | Preprocessing, Training, Inference, Tests | ✅ Complete & passing (13/13 tests) |
| **API Endpoints** | Validate, Classify, Segment, Review, Audit | ✅ Complete & verified |
| **Frontend Viewer** | Dual-modal, opacity slider, audit history | ✅ Complete React application |
| **Data Governance** | Provenance, schema, leakage check | ✅ Fully documented in `DATA.md` |
| **Full GPU Checkpoints** | Trained on 112,120 ChestX-rays & MSD CTs | ⏳ Baseline scaffold verified; full GPU training pending |
| **Report (`report.pdf`)** | Written capstone report | ⏳ Outline prepared; requires full GPU figures |
| **Slides (`presentation.pdf`)** | 10-minute presentation slides | ⏳ Structure documented |
| **Demo Video** | Recorded demonstration walk-through | ⏳ Pending video recording |
