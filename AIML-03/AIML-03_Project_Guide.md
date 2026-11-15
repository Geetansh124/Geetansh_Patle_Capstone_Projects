# AIML-03 — Large-Scale Personalized Recommendation Engine

## Project information

- **Track:** AI / Machine Learning
- **Batch start:** 29 October 2026
- **Deadline:** 30 November 2026
- **Estimated effort:** 14 days
- **Goal:** Build a two-stage recommender system: fast candidate retrieval followed by precise ranking.

## 1. Problem statement

Online platforms often have millions of items and cannot score every item for every user. This project implements a production-style pipeline that first retrieves a few hundred relevant candidates and then ranks them using richer user, item, and interaction features.

The system will use MovieLens 25M as the primary dataset. All reported metrics must come from a documented, reproducible evaluation run. No online click-through rate (CTR) will be claimed unless a real online experiment is conducted.

## 2. Target architecture

```text
MovieLens 25M
     |
     v
Timestamp-aware cleaning and chronological split
     |
     v
Spark feature tables ----> Feature artifacts / manifests
     |
     +--> TFRS two-tower retrieval --> FAISS top-N candidates
                                      |
                                      v
                         Wide & Deep ranking model
                                      |
                                      v
                      Redis cache --> Recommendation API

Airflow: scheduled retraining and evaluation
Triton: optional model serving configuration
```

## 3. Technology choices

- **Data processing:** Apache Spark and Python
- **Retrieval:** TensorFlow Recommenders two-tower model
- **Approximate nearest-neighbour search:** FAISS
- **Ranking:** Wide & Deep neural ranker
- **Caching:** Redis
- **Workflow orchestration:** Apache Airflow
- **Serving configuration:** NVIDIA Triton Inference Server
- **API/demo:** FastAPI or an equivalent lightweight Python service
- **Packaging:** Docker Compose where practical

## 4. Implementation phases

### Phase 1 — Repository and environment setup

1. Create the repository and project folders.
2. Add `requirements.txt` or `environment.yml`, Docker files, and a reproducibility note.
3. Keep datasets, model checkpoints, FAISS indexes, and secrets outside Git.
4. Add `.gitignore`, configuration templates, and a small synthetic fixture for local tests.
5. Record Python, Spark, TensorFlow, TFRS, FAISS, Redis, Airflow, and Triton versions.

**Exit criteria:** A clean checkout can install dependencies and run a smoke test without downloading the full dataset.

### Phase 2 — Data ingestion and validation

1. Download MovieLens 25M from its official source and record the URL and checksum.
2. Validate required columns, duplicate rows, missing values, timestamp ranges, and rating bounds.
3. Sort each user’s interactions by timestamp.
4. Split each user chronologically into train, validation, and test partitions.
5. Ensure no future interaction leaks into training features.
6. Save only lightweight manifests and schema documentation in the repository.

**Required checks:** user/item overlap, timestamp boundary checks, duplicate detection, and a leakage report.

### Phase 3 — Spark feature engineering

Create documented feature tables for:

- user history and interaction counts;
- item rating count and average rating;
- time-based features;
- user-item recency features;
- genre and item metadata;
- negative or non-interacted examples for ranking evaluation.

The feature job must be deterministic for a fixed input snapshot and configuration.

### Phase 4 — Two-tower retrieval model

1. Encode users from IDs and available user-history features.
2. Encode movies from IDs, genres, and metadata.
3. Train the towers with TensorFlow Recommenders retrieval objectives.
4. Export item embeddings from the training snapshot.
5. Build a FAISS index over item embeddings.
6. Retrieve top-N candidates for each validation/test user.

Measure **Recall@K** for retrieval separately from the final ranking metrics.

### Phase 5 — Wide & Deep ranking model

1. Combine user, item, interaction, retrieval-score, and context features.
2. Train the ranker only on the training partition.
3. Compare it with a simple baseline, such as popularity ranking.
4. Evaluate both rankers on the same held-out test split.
5. Report Recall@K and NDCG@K for clearly defined K values.

Do not tune thresholds on the test set. Save the configuration and random seeds for every run.

### Phase 6 — Serving and fallbacks

Implement an API endpoint such as:

```text
GET /recommendations/{user_id}?k=10
```

The service must:

- validate the user ID and requested K;
- use Redis for repeat-request caching;
- retrieve candidates through FAISS;
- rank candidates with the ranking model;
- remove already-seen items when configured;
- return a popularity fallback for unknown users;
- return a content or popularity fallback for unknown items;
- expose a health endpoint;
- avoid returning private training data.

### Phase 7 — Airflow and Triton configuration

Create an Airflow DAG that represents:

```text
validate data -> build Spark features -> train retrieval -> build FAISS
-> train ranker -> evaluate -> publish versioned artifacts
```

Add retries, logging, configuration parameters, and a failure notification placeholder. Provide a Triton model repository configuration, even if full GPU serving is documented as an optional deployment path.

### Phase 8 — Testing and evidence

Test:

- chronological split boundaries;
- leakage detection;
- deterministic preprocessing;
- retrieval index loading;
- ranker checkpoint loading;
- invalid API inputs;
- cold-start behavior;
- Redis cache hit and miss behavior;
- Airflow DAG import;
- Docker Compose startup;
- metric calculation against saved predictions.

Save test output, architecture diagrams, API examples, and screenshots in `screenshots/` and `data_documentation/`. Use only real outputs from completed runs.

## 5. Required deliverables

- `code/` — preprocessing, training, evaluation, serving, tests, and configuration.
- `data_documentation/` — source, license, schema, checksums, split policy, and leakage report.
- `screenshots/` — API response, dashboard/demo, metrics, and pipeline evidence.
- `README.md` — setup, architecture, commands, and reproducibility steps.
- `report.pdf` — problem, design, implementation, measured results, limitations, and reflection.
- `presentation.pdf` — approximately 10 minutes of slides.
- `demo_video_link.txt` — keep a clearly labelled pending entry until recording exists.

## 6. Results table template

| Model | Split | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Run ID |
|---|---|---:|---:|---:|---:|---|
| Popularity baseline | Test | pending | pending | pending | pending | pending |
| Two-stage model | Test | pending | pending | pending | pending | pending |

Replace every `pending` value only after the corresponding run has completed and its artifact is saved.

## 7. Submission checklist

- [ ] Repository is public and contains no secrets, datasets, or large checkpoints.
- [ ] A fresh environment can run the smoke test.
- [ ] Data source, versions, commands, and random seeds are documented.
- [ ] Chronological split and leakage checks pass.
- [ ] Retrieval and ranking metrics are reproducible.
- [ ] API, cache, fallback, Airflow, and Docker tests pass.
- [ ] Report and presentation match the saved results.
- [ ] Demo link is real, accessible, and labelled correctly.
- [ ] GitHub repository link is ready for Nexvia submission.

## 8. Limitations

MovieLens is an offline benchmark and does not represent real production traffic, business constraints, or user diversity. Offline ranking metrics do not prove improved user satisfaction. Cold-start performance, popularity bias, and temporal drift must be discussed explicitly.
