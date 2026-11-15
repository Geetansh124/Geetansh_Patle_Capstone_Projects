# AIML-03 — Large-Scale Personalized Recommendation Engine

**Two-Stage Retrieval & Neural Ranking Architecture**  
**Lead Author:** Geetansh Patle  
**Track:** AI / Machine Learning Capstone  
**Status:** Runnable baseline, serving API, Redis caching, Airflow DAG, and verification suite complete.

---

## 1. System Architecture

```text
               [ MovieLens 25M Dataset ]
                         |
                         v
       [ Chronological Split & Leakage Check ]
            (Per-user 80/20 temporal cutoff)
                         |
        +----------------+----------------+
        |                                 |
        v                                 v
[ Spark Feature Tables ]       [ Baseline Popularity & ]
 (User histories, items)       [ Genre Personalization ]
        |                                 |
        v                                 v
[ TFRS Two-Tower Model ]       [ Offline Holdout Eval ]
  (Embedding dim: 64)           (Recall@K & NDCG@K)
        |                                 |
        v                                 v
[ FAISS Item Index ]           [ Airflow Orchestration ]
  (Cosine inner product)         (dags/recommender_pipeline_dag.py)
        |                                 |
        +----------------+----------------+
                         |
                         v
        [ Wide & Deep Ranking Model ]
         (Keras SavedModel / Triton)
                         |
                         v
        [ FastAPI Serving Gateway ] <---> [ Redis Cache ]
          (/recommend/{userId}?k=10)        (TTL: 300s)
                         |
                         v
       [ Cold-Start & Popularity Fallbacks ]
```

---

## 2. Directory Layout

- `code/src/api.py`: FastAPI serving service with Redis caching, candidate retrieval, and cold-start fallback.
- `code/src/pipeline.py`: Chronological split, popularity baseline, genre personalization, Spark feature generator, TFRS retrieval, and Wide & Deep ranker.
- `code/dags/recommender_pipeline_dag.py`: Production Apache Airflow DAG for scheduled data validation, training, and gating.
- `code/triton_models/wide_deep/config.pbtxt`: NVIDIA Triton Inference Server configuration.
- `code/scripts/generate_synthetic_fixtures.py`: One-command synthetic MovieLens dataset generator and runner.
- `code/tests/`: Automated unit and integration test suite (`pytest`).
- `data_documentation/DATA.md`: Detailed schema, chronological partitioning formulation, and offline vs. online disclosures.

---

## 3. Quickstart & Verification

### Local Setup
```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Run automated test suite
pytest code/tests

# 3. Generate synthetic MovieLens fixtures and run baseline pipeline
python code/scripts/generate_synthetic_fixtures.py
```

### Running Services with Docker Compose
```bash
# Launch Redis and Recommendation API
docker compose -f code/docker-compose.yml up --build

# Or with Airflow and Triton profiles enabled:
docker compose -f code/docker-compose.yml --profile orchestration --profile gpu-serving up
```

### Querying Serving API
```bash
# Get recommendations for user 1 (K=10)
curl http://localhost:8000/recommend/1?k=10

# Inspect service health and cache status
curl http://localhost:8000/health
```

---

## 4. Truthful Status & Deliverable Checklist

| Deliverable | Required Spec | Current Status |
| :--- | :--- | :--- |
| **Pipeline & Baseline** | Chronological split, Popularity, Genre-aware | ✅ Complete & verified |
| **Two-Stage Architecture** | TFRS Two-Tower + FAISS + Wide & Deep | ✅ Code written & pipeline-ready |
| **Serving API** | FastAPI + Redis Caching + Cold Start | ✅ Complete & verified (5/5 tests) |
| **Orchestration** | Airflow Retraining DAG | ✅ Complete in `dags/` |
| **GPU Serving** | Triton Inference Model Config | ✅ Configured in `triton_models/` |
| **Data Governance** | Provenance, schema, leakage check | ✅ Fully documented in `DATA.md` |
| **25M Scale GPU Run** | Full training run on all 25M ratings | ⏳ Local baseline verified; 25M GPU run pending |
| **Report (`report.pdf`)** | Written capstone report | ⏳ Outline prepared; pending full scale metrics |
| **Slides (`presentation.pdf`)** | 10-minute presentation slides | ⏳ Structure documented |
| **Demo Video** | Recorded demonstration walk-through | ⏳ Pending video recording |
