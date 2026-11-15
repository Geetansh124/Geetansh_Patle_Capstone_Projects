# AIML-02 — AI-Assisted Medical Imaging and Diagnosis System

## Project information

- **Track:** AI / Machine Learning
- **Batch start:** 29 October 2026
- **Deadline:** 30 November 2026
- **Estimated effort:** 14 days
- **Goal:** Build a research and education platform that accepts medical images, runs deep-learning inference, shows explanation overlays, and records human review decisions.

> **For education only — not for clinical use.** This system is a portfolio and capstone project. It must not be used to diagnose, triage, or treat patients.

## 1. Problem statement

The platform will demonstrate an end-to-end workflow for medical-image analysis. It will support an X-ray classification task and a separate 3D lung-segmentation task, with model explanations and an auditable review interface.

The classification dataset is NIH ChestX-ray14. The segmentation dataset is Medical Segmentation Decathlon Task06 lung CT data. Dataset labels, demographic coverage, acquisition differences, and known limitations must be documented rather than hidden.

## 2. Target architecture

```text
Orthanc DICOM server / image upload
              |
              v
      Validation and preprocessing
          |                 |
          v                 v
   X-ray classifier      3D U-Net segmenter
     PyTorch transfer       MONAI + PyTorch
          |                 |
          +-------> Grad-CAM / overlay generation
                              |
                              v
        Inference API + React education viewer
                              |
             approve/reject + audit event store
```

## 3. Technology choices

- **Deep learning:** PyTorch
- **Medical imaging framework:** MONAI
- **DICOM server:** Orthanc
- **Explanations:** Grad-CAM for the classifier; transparent segmentation overlays for the 3D model
- **Dataset/version management:** DVC-managed manifests and metadata
- **Frontend:** React
- **Packaging:** Docker and Docker Compose
- **API:** FastAPI or an equivalent Python service

## 4. Safety and ethics requirements

The education-only warning must appear in the viewer, API documentation, screenshots, README, report, and presentation. The system must:

- avoid claiming diagnosis or clinical accuracy;
- show confidence as model confidence, not medical certainty;
- display limitations beside results;
- avoid exposing patient identifiers in logs or screenshots;
- use de-identified/public data only;
- record review actions without pretending they are clinical decisions;
- document label noise, missing findings, bias, domain shift, and false-positive/false-negative risks.

## 5. Implementation phases

### Phase 1 — Repository and environment setup

1. Create the project structure and Python environment.
2. Add PyTorch, MONAI, DVC, FastAPI, Orthanc, React, and testing dependencies.
3. Add Docker Compose for the API, viewer, and Orthanc where practical.
4. Keep datasets, DICOM files, model weights, and secrets outside Git.
5. Add a small synthetic test image/volume fixture for local tests.
6. Record CUDA, GPU, Python, PyTorch, MONAI, and OS versions for Kaggle training.

**Exit criteria:** The API and viewer start locally with synthetic data and show the education-only notice.

### Phase 2 — Data acquisition and documentation

For NIH ChestX-ray14 and MSD Task06:

1. Record the official source, access conditions, version, and checksum.
2. Document file formats, labels, resolution, spacing, and missing-value rules.
3. Create DVC-managed manifests without committing the data itself.
4. Identify patient or study identifiers where available.
5. Create patient-safe train/validation/test splits whenever identifiers permit.
6. Run checks for duplicate studies and cross-split identity leakage.

Do not merge the classification and segmentation datasets into one evaluation claim; they serve different tasks.

### Phase 3 — Preprocessing pipelines

#### X-ray classification

- decode and validate images;
- normalize intensity;
- resize to the model input size;
- define augmentation only for training;
- handle multi-label findings explicitly;
- document class imbalance and missing-label assumptions.

#### 3D lung segmentation

- load CT volumes and masks;
- verify orientation and spacing;
- resample according to a documented policy;
- clip or normalize intensity values;
- crop or pad volumes deterministically;
- preserve a mapping back to the original study space for overlays.

### Phase 4 — Classification model and Grad-CAM

1. Start with a transfer-learning backbone in PyTorch.
2. Train using a patient-safe split.
3. Track loss and task-appropriate metrics such as AUROC, sensitivity, specificity, and F1 where valid.
4. Save the model configuration, checkpoint hash, and evaluation outputs.
5. Generate Grad-CAM overlays for representative true-positive, false-positive, and uncertain examples.
6. Explain that Grad-CAM highlights influential regions; it does not prove causal medical evidence.

### Phase 5 — MONAI 3D U-Net segmentation

1. Build the training dataset from the documented Task06 manifest.
2. Train a MONAI 3D U-Net on Kaggle GPU.
3. Evaluate with Dice and IoU, with per-case and aggregate values where feasible.
4. Save predictions and overlay examples from the held-out split.
5. Test checkpoint loading independently from training.

### Phase 6 — Orthanc and inference API

Implement an ingestion path that:

- accepts supported DICOM studies;
- validates file type and required metadata;
- rejects malformed or unsupported inputs safely;
- strips or masks identifiers in demo logs;
- preprocesses the study consistently;
- runs the selected model;
- returns class scores, segmentation metadata, overlays, model version, and a clear education-only warning;
- never states that a patient has or does not have a disease.

Suggested endpoints:

```text
GET  /health
POST /studies/validate
POST /inference/classification
POST /inference/segmentation
POST /review-events
```

### Phase 7 — React viewer and audit events

The viewer should provide:

- scan/study selection;
- image or slice navigation;
- overlay visibility and opacity controls;
- model confidence display with explanatory text;
- model/version and timestamp;
- approve/reject controls for demonstration review;
- a visible audit-event history;
- an always-visible “For education only — not for clinical use” notice.

Approve/reject is a demo review action, not a clinical diagnosis. Audit events should include the study pseudonym, model version, action, timestamp, and optional comment, but never unnecessary patient identifiers.

### Phase 8 — Testing and evidence

Test:

- manifest and checksum validation;
- patient-level split boundaries;
- leakage detection;
- malformed DICOM handling;
- unsupported modality handling;
- checkpoint loading;
- classification and segmentation output shapes;
- Grad-CAM generation;
- overlay rendering;
- API input validation;
- approve/reject audit persistence;
- education-only warning visibility;
- Docker Compose startup;
- report figures against saved evaluation results.

Save actual test logs, metrics, screenshots, and sample API responses. Do not invent clinical or performance results.

## 6. Required deliverables

- `code/` — preprocessing, training, inference API, viewer source, tests, and configuration.
- `data_documentation/` — dataset sources, licenses, manifests, split policy, label limitations, and ethics notes.
- `screenshots/` — viewer, overlay, confidence display, Orthanc/API flow, and audit events.
- `README.md` — setup, safety notice, architecture, commands, and reproducibility steps.
- `report.pdf` — problem, design, datasets, implementation, measured results, limitations, and reflection.
- `presentation.pdf` — approximately 10 minutes of slides.
- `demo_video_link.txt` — keep a clearly labelled pending entry until recording exists.

## 7. Results table template

### Classification

| Model | Split | AUROC | Sensitivity | Specificity | F1 | Run ID |
|---|---|---:|---:|---:|---:|---|
| Transfer-learning classifier | Test | pending | pending | pending | pending | pending |

### Segmentation

| Model | Split | Dice | IoU | Cases | Run ID |
|---|---|---:|---:|---:|---|
| MONAI 3D U-Net | Test | pending | pending | pending | pending |

Replace `pending` only with values produced by completed, saved evaluation runs.

## 8. Submission checklist

- [ ] Public GitHub repository contains code and documentation but no private data, DICOM files, or secrets.
- [ ] Dataset sources, licenses, versions, and manifests are documented.
- [ ] Patient-safe split and leakage checks pass.
- [ ] Training can be reproduced on the documented Kaggle GPU setup.
- [ ] Checkpoint loading and inference tests pass on local CPU where practical.
- [ ] Orthanc ingestion, malformed-input handling, and API validation pass.
- [ ] React viewer renders scan/overlay, confidence, review action, and audit history.
- [ ] Education-only notice appears throughout the product and deliverables.
- [ ] Report and presentation use only real measured results.
- [ ] Demo link is real and accessible before submission.
- [ ] GitHub repository link is ready for Nexvia submission.

## 9. Limitations

Public datasets may contain noisy labels, limited demographics, acquisition bias, and incomplete clinical context. A model can fail under distribution shift and can produce misleading explanations. Performance on a held-out public dataset does not establish clinical validity, safety, regulatory approval, or usefulness in a hospital workflow.
