# Production stack implementation sequence

1. Spark: materialize timestamped interactions, user histories, item popularity/category features; persist Parquet and record schema.
2. Retrieval: TensorFlow Recommenders two-tower; export item embeddings; evaluate Recall@K against the fixed temporal holdout.
3. FAISS: build/rebuild an index from versioned item embeddings; test unknown IDs and empty catalog.
4. Ranking: train Wide & Deep and a simple logistic baseline on train-only positives plus documented sampled negatives; evaluate NDCG@K on identical candidates/split.
5. API/cache: FastAPI endpoint accepts user ID and K; Redis cache keys include model/index version; cold-start returns popular/content candidates.
6. Operations: Airflow DAG validates inputs, trains, evaluates, and promotes only on threshold; Triton config serves exported ranker. Local compose is for integration/demo, not production security.
7. Reporting: no online CTR claim without a real, consented, instrumented A/B experiment.
