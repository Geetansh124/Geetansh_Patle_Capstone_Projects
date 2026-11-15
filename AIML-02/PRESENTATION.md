# AIML-02 Capstone Presentation Deck
## AI-Assisted Medical Imaging and Diagnosis System

**Presenter:** Geetansh Patle  
**Track:** AI / Machine Learning Capstone  
**Target Duration:** 10 Minutes (10 Slides)

---

### Slide 1: Title & Overview
- **Title:** AI-Assisted Medical Imaging and Diagnosis System
- **Subtitle:** An Educational Platform for Multimodal Radiographic Inference and Review
- **Presenter:** Geetansh Patle
- **Key Takeaway:** End-to-end medical deep learning pipeline combining transfer learning, 3D volumetric segmentation, Grad-CAM interpretability, and human audit logging.
- **Speaker Notes (1 min):** "Welcome everyone. Today I'm presenting AIML-02, a medical vision prototype developed to bridge deep learning research with clinical audit workflows while enforcing strict patient-safe validation."

---

### Slide 2: Problem Statement & Clinical Challenges
- High clinical burnout and increasing medical imaging backlogs.
- "Black-box" deep learning models lack clinician trust.
- Widespread patient identity leakage in academic datasets distorts real-world performance.
- Severe regulatory requirements: systems must fail-closed and retain traceable human review events.
- **Speaker Notes (1 min):** "Machine learning in healthcare is useless without explainability and safety. In this project, we prioritize zero patient leakage and verifiable human-in-the-loop auditability over uncalibrated performance claims."

---

### Slide 3: Ethical Guardrails & Safety Policy
- **Education Only:** Bounded as an educational and research prototype.
- **Calibrated Scores:** Probabilities represent model activation statistics, not diagnostic certainty.
- **Privacy Enforcement:** Zero PHI ingested; all pipelines operate on de-identified cohort identifiers.
- **Immutable Audit Trail:** Append-only JSONL recording every human verification action.
- **Speaker Notes (1 min):** "A core architectural pillar is our safety boundary. Every screen, API response, and documentation file carries explicit warnings, and all review decisions are permanently audited."

---

### Slide 4: System Architecture
- **PACS Gateway:** Orthanc DICOM server for study ingestion and instance preview.
- **Inference Service:** FastAPI microservice handling validation, classification, and segmentation.
- **Dual-Model Processing:**
  - 2D DenseNet-121 for Chest Radiographs.
  - 3D MONAI U-Net for Thoracic CT Scans.
- **Interactive UI:** React 19 application deployed live on Vercel.
- **Speaker Notes (1 min):** "Here is the full system architecture, connecting hospital-standard Orthanc DICOM servers with our PyTorch and MONAI inference backends and our deployed React viewer."

---

### Slide 5: Data Engineering & Zero-Leakage Splitting
- **NIH ChestX-ray14:** 112,120 frontal chest radiographs across 14 pathologies.
- **MSD Task06:** 3D helical CT scans in NIfTI format.
- **The Grouped Split Rule:** Studies belonging to the same `Patient ID` are never split across train and test sets.
- **Verification:** Automated leakage scanner confirms 0 overlapping patient IDs.
- **Speaker Notes (1 min):** "In ChestX-ray14, patients undergo repeated follow-up scans. Splitting randomly causes massive test leakage. We enforce patient-grouped splitting with zero overlap."

---

### Slide 6: Chest X-ray Classification & Grad-CAM
- **Backbone:** DenseNet-121 pre-trained on ImageNet with multi-label binary cross-entropy loss.
- **Grad-CAM Saliency:**
  - Gradient backpropagation targeting DenseBlock 4.
  - Class-specific spatial activation map upsampled to 224×224.
  - Heatmap visual overlay with real-time opacity slider in React viewer.
- **Speaker Notes (1 min):** "For classification, we extract spatial gradients from the final dense block to produce Grad-CAM heatmaps, providing visual feature attribution for every predicted finding."

---

### Slide 7: 3D Volumetric Lung CT Segmentation
- **Framework:** MONAI (Medical Open Network for AI).
- **Preprocessing:** Standardized RAS orientation, 1.5×1.5×2.0 mm voxel resampling, [-1000, 400] HU intensity windowing.
- **Network:** 3D Residual U-Net with DiceCELoss optimization.
- **Output:** Volumetric lesion contour masks with slice navigation slider.
- **Speaker Notes (1 min):** "For volumetric CT, we normalize voxel spacing to 1.5mm and train a 3D U-Net using balanced foreground patch sampling, generating spatial lesion contours."

---

### Slide 8: Interactive React Viewer & Audit Trail
- **Live Vercel URL:** [https://frontend-ten-pi-61.vercel.app](https://frontend-ten-pi-61.vercel.app)
- **Features:**
  - Dual modality switcher (Chest X-ray vs. 3D CT).
  - Dynamic Grad-CAM heatmap opacity slider.
  - Ranked model findings with probability indicators.
  - Verification review buttons (Accept / Flag for Re-evaluation) linked to immutable audit logs.
- **Speaker Notes (1 min):** "Here is our live deployed application on Vercel. Reviewers can upload scans, adjust explanation overlays, and record governance decisions in real time."

---

### Slide 9: Automated Verification & Testing
- **Test Framework:** `pytest` running 13 comprehensive test modules.
- **Coverage Areas:**
  - DICOM modality validation (DX/CR/CT support).
  - Patient split leakage detection.
  - Grad-CAM and segmentation output tensor shapes.
  - API fail-closed behavior without Orthanc.
  - Review event persistence across restarts.
- **Speaker Notes (1 min):** "Our codebase passes 100% of our test suite in 0.92 seconds, guaranteeing that safety rules and API contracts are fully honored."

---

### Slide 10: Limitations, Conclusion & Next Steps
- **Limitations:** Report-mined label noise (10-15%), lack of multi-scanner generalization, Grad-CAM attribution correlation vs. causation.
- **Future Work:** Federated multi-center training, radiologist uncertainty calibration, DICOM Structured Reporting (SR) export.
- **Conclusion:** A robust, verified, and transparent foundation for educational medical AI.
- **Speaker Notes (1 min):** "In conclusion, AIML-02 demonstrates how explainability and patient-safe engineering can be integrated into medical software. Thank you, and I look forward to your questions."
