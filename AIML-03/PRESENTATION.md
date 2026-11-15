# AIML-03 Capstone Presentation Deck
## Large-Scale Personalized Recommendation Engine

**Presenter:** Geetansh Patle  
**Track:** AI / Machine Learning Capstone  
**Target Duration:** 10 Minutes (10 Slides)

---

### Slide 1: Title & Overview
- **Title:** Large-Scale Personalized Recommendation Engine
- **Subtitle:** Two-Stage Candidate Retrieval and Deep Neural Ranking on MovieLens 25M
- **Presenter:** Geetansh Patle
- **Key Architecture:** TFRS Two-Tower Retrieval + FAISS Indexing + Wide & Deep Ranking + Redis Serving + Airflow Orchestration.
- **Speaker Notes (1 min):** "Hello everyone. Today I'm presenting AIML-03, a production-style two-stage recommendation engine designed to deliver personalized recommendations over large-scale catalogs in sub-50ms latency."

---

### Slide 2: The Core Problem: The Scale Bottleneck
- Massive item catalogs ($>10^5$ items) make end-to-end deep neural scoring impossible within a 50ms SLA.
- Naive ranking over $N$ items for $U$ users requires $O(U \times N)$ complex network forward passes.
- Solution: Decouple candidate generation into two distinct stages:
  - **Stage 1 (Retrieval):** Fast candidate reduction ($100,000 \to 100$).
  - **Stage 2 (Ranking):** Heavyweight non-linear feature scoring ($100 \to 10$).
- **Speaker Notes (1 min):** "When platforms have millions of items, scoring every movie for every user is impossible in real time. We solve this by decoupling retrieval from ranking."

---

### Slide 3: Temporal Split & Zero-Leakage Protocol
- **Random K-Fold is Invalid:** Randomly splitting interactions allows models to train on future ratings to predict past ones (fatal time leakage).
- **Chronological Rule:** Interactions are sorted by Unix timestamp for every individual user.
- **80/20 Boundary:** First 80% of interactions $\to$ Training; Latest 20% $\to$ Held-out test set.
- **Candidate Pool:** Recommenders must rank unseen catalog items; items already consumed in training are filtered.
- **Speaker Notes (1 min):** "Time matters in recommender systems. We enforce strict chronological partitioning within each user's timeline so future data never leaks into past features."

---

### Slide 4: System Architecture
- **Offline Pipeline:** Spark feature tables $\to$ TFRS embedding export $\to$ FAISS vector indexing $\to$ Wide & Deep training.
- **Online Serving:** FastAPI gateway querying Redis cache (TTL 300s) with sub-5ms response time on cache hits.
- **Workflow Orchestration:** Airflow DAG enforcing weekly retraining and metric gating.
- **Inference Server:** NVIDIA Triton model repository configuration (`config.pbtxt`).
- **Speaker Notes (1 min):** "This slide shows our complete architecture, from offline Spark data processing and TensorFlow training to live Redis caching and FastAPI serving."

---

### Slide 5: Stage 1 — Two-Tower Retrieval & FAISS
- **Two-Tower Architecture:**
  - User Tower: Embeds user IDs and interaction histories into $\mathbb{R}^{64}$.
  - Item Tower: Embeds movie IDs and genres into $\mathbb{R}^{64}$.
- **Objective:** In-batch sampled softmax loss maximizing inner-product cosine similarity.
- **FAISS Indexing:**
  - $L_2$-normalized item vectors indexed into `IndexFlatIP`.
  - Retrieves top 100 candidate items in $<1$ ms.
- **Speaker Notes (1 min):** "Stage 1 uses dual deep towers to map users and items into a shared 64-dimensional space, followed by FAISS inner-product vector indexing."

---

### Slide 6: Stage 2 — Wide & Deep Neural Ranking
- **The Ranking Objective:** Discriminate subtle feature interactions within the top-100 retrieved candidates.
- **Wide Component:** Memorizes cross-product feature interactions via linear model.
- **Deep Component:** Generalizes to unseen user-item pairs via 3 dense layers (64 $\to$ 32 $\to$ 1) with ReLU activations.
- **Output:** Sigmoid click/interaction probability for final Top-K sorting.
- **Speaker Notes (1 min):** "Stage 2 re-ranks the 100 candidates using a Wide & Deep architecture, combining memorization of frequent pairs with deep generalization."

---

### Slide 7: Serving Microservice, Redis Caching & Fallbacks
- **FastAPI Endpoint:** `GET /recommend/{user_id}?k=10`
- **Redis Cache Layer:**
  - High-performance caching key: `rec:u{id}:k{k}` (TTL: 300s).
  - Eliminates redundant inference on repeat requests.
- **Cold-Start Fallbacks:**
  - Unseen users receive global popularity rankings excluding seen items.
  - Unseen items receive genre-affinity heuristic scoring.
- **Speaker Notes (1 min):** "Our serving layer uses Redis to cache recommendations for 5 minutes, and includes robust fallbacks so cold-start users never encounter errors."

---

### Slide 8: Automated Airflow Orchestration DAG
- **DAG Workflow:**
  `Validate Data -> Build Spark Features -> Train Retrieval -> Build FAISS -> Train Ranker -> Evaluate -> Publish Artifacts`
- **Quality Gate:** Automated check verifies holdout Recall@10 exceeds acceptance threshold before deploying new models.
- **Production Safety:** Auto-retry (2 attempts), error notification hooks, weekly schedule.
- **Speaker Notes (1 min):** "We orchestrated the entire retraining pipeline with an Apache Airflow DAG that gates artifact deployment on holdout evaluation metrics."

---

### Slide 9: Automated Verification & Testing
- **Test Framework:** `pytest` covering 11 dedicated test cases.
- **Pass Rate:** 100% across all modules in 0.92s.
- **Tests Include:**
  - Chronological split disjointness and timestamp monotonic order.
  - Ranking metrics bounded in $[0.0, 1.0]$.
  - Cache hit/miss and invalid parameter rejection ($k \le 0, k > 100$).
  - Airflow DAG syntax and task dependency compilation.
- **Speaker Notes (1 min):** "Every component is covered by automated unit and integration tests, ensuring all boundary conditions and pipeline invariants hold."

---

### Slide 10: Limitations, Conclusion & Next Steps
- **Honest Disclosures:**
  - Offline Recall and NDCG measure historical retrieval; they do NOT prove online Click-Through Rate (CTR).
  - Popularity bias requires exploration (e.g. contextual bandits) in live production.
- **Conclusion:** A high-throughput, leakage-free, and production-tested recommendation system.
- **Speaker Notes (1 min):** "In conclusion, AIML-03 provides a complete blueprint for large-scale recommendation. Thank you for your time, and I am happy to answer any questions."
