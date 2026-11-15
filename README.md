# Geetansh Patle — Capstone Projects Portfolio

Submission package for Nexvia AI/ML Capstones: **AIML-02** and **AIML-03**.

---

## Portfolio Summary

| Project | Domain | Architecture | Key Technologies | Test Status |
| :--- | :--- | :--- | :--- | :--- |
| **[AIML-02](./AIML-02/)** | Medical Imaging & Diagnosis | DenseNet-121 Classifier + Grad-CAM + MONAI 3D U-Net | PyTorch, MONAI, FastAPI, Orthanc, React | 13/13 passing |
| **[AIML-03](./AIML-03/)** | Large-Scale Recommendation | Two-Tower Candidate Retrieval + Wide & Deep Ranker | TensorFlow, TFRS, FAISS, Redis, Airflow, Triton | 10/10 passing |

---

## Ethical & Research Disclaimers

1. **AIML-02 (Medical Imaging):** **For research and education only — not for clinical use.** The system is designed for portfolio and educational demonstration. It must never be used to diagnose, triage, or guide clinical treatment.
2. **AIML-03 (Recommendation Engine):** All metrics (Recall@K, NDCG@K) represent **offline historical holdout evaluation** on MovieLens. No online Click-Through Rate (CTR) or commercial conversion is claimed.

---

## Global Verification & Smoke Test

To verify both project pipelines from repository root:

```bash
# Run complete test suite across both projects (23 tests)
pytest

# Generate synthetic fixtures and baseline artifacts
python AIML-02/code/scripts/generate_synthetic_fixtures.py
python AIML-03/code/scripts/generate_synthetic_fixtures.py
```
