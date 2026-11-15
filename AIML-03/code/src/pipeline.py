"""Time-aware MovieLens popularity baseline and ranking metrics."""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def chronological_split(ratings: pd.DataFrame, fraction: float = 0.8):
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    required = {"userId", "movieId", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings missing columns: {sorted(required - set(ratings.columns))}")
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).copy()
    # Per-user cutoff prevents future interactions from leaking into training.
    ranks = ratings.groupby("userId").cumcount()
    sizes = ratings.groupby("userId")["movieId"].transform("size")
    # Keep at least one interaction in train and one in test for users with >=2 rows.
    cutoffs = (sizes * fraction).apply(int).clip(lower=1)
    # Single-interaction users remain in train (no within-user holdout is possible).
    cutoffs = cutoffs.where(sizes > 1, 1)
    train_mask = ranks < cutoffs
    return ratings[train_mask].copy(), ratings[~train_mask].copy()


def recommend(train: pd.DataFrame, user_id: int, k: int = 10):
    if k < 1:
        raise ValueError("k must be at least 1")
    seen = set(train.loc[train.userId == user_id, "movieId"])
    popularity = train.groupby("movieId").size().sort_values(ascending=False)
    return [int(mid) for mid in popularity.index if mid not in seen][:k]


def ranking_metrics(test: pd.DataFrame, train: pd.DataFrame, k: int = 10):
    truth = test.groupby("userId").movieId.apply(set)
    recalls, ndcgs = [], []
    for user, relevant in truth.items():
        recs = recommend(train, int(user), k)
        hits = [int(item in relevant) for item in recs]
        recalls.append(sum(hits) / max(1, len(relevant)))
        dcg = sum(hit / __import__("math").log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / __import__("math").log2(i + 2) for i in range(min(len(relevant), k)))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {"recall_at_k": sum(recalls) / max(1, len(recalls)), "ndcg_at_k": sum(ndcgs) / max(1, len(ndcgs)), "k": k, "users": len(recalls)}


def compare_popularity_and_personalized(train: pd.DataFrame, test: pd.DataFrame, movies: pd.DataFrame, k: int = 10):
    """Compare popularity with genre-personalized ranking over one shared candidate set."""
    genres = movies.set_index("movieId").genres.fillna("").to_dict()
    catalog = set(map(int, movies.movieId))
    pop = train.groupby("movieId").size().to_dict()
    user_genres = {}
    for uid, rows in train.merge(movies[["movieId", "genres"]], on="movieId", how="left").groupby("userId"):
        counts = {}
        for value in rows.genres.fillna(""):
            for genre in value.split("|"):
                if genre and genre != "(no genres listed)": counts[genre] = counts.get(genre, 0) + 1
        user_genres[int(uid)] = counts
    truths = test.groupby("userId").movieId.apply(lambda x: set(map(int, x)))
    totals = {"popularity": [0.0, 0.0], "genre_personalized": [0.0, 0.0]}
    users = 0
    for uid, relevant in truths.items():
        seen = set(map(int, train.loc[train.userId == uid, "movieId"]))
        candidates = catalog - seen
        if not candidates: continue
        profile = user_genres.get(int(uid), {})
        ranked = {
            "popularity": sorted(candidates, key=lambda i: (-pop.get(i, 0), i))[:k],
            "genre_personalized": sorted(candidates, key=lambda i: (-sum(profile.get(g, 0) for g in genres.get(i, "").split("|")), -pop.get(i, 0), i))[:k],
        }
        users += 1
        for name, recs in ranked.items():
            hits = [int(i in relevant) for i in recs]
            recall = sum(hits) / max(1, len(relevant))
            dcg = sum(hit / np.log2(j + 2) for j, hit in enumerate(hits))
            ideal = sum(1 / np.log2(j + 2) for j in range(min(len(relevant), k)))
            totals[name][0] += recall; totals[name][1] += dcg / ideal if ideal else 0
    return {name: {"recall_at_k": vals[0]/max(users,1), "ndcg_at_k": vals[1]/max(users,1), "k":k, "users":users, "ctr":None} for name, vals in totals.items()}


def build_spark_features(ratings_path: Path, output_dir: Path):
    """Write scalable interaction and item feature tables; Spark is optional for local baseline."""
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    spark = SparkSession.builder.appName("AIML03FeatureBuild").getOrCreate()
    ratings = spark.read.option("header", True).option("inferSchema", True).csv(str(ratings_path))
    window = Window.partitionBy("userId").orderBy("timestamp", "movieId")
    interactions = ratings.withColumn("user_event_index", F.row_number().over(window))
    interactions.write.mode("overwrite").parquet(str(output_dir / "interactions"))
    items = ratings.groupBy("movieId").agg(F.count("*").alias("popularity"), F.avg("rating").alias("mean_rating"))
    items.write.mode("overwrite").parquet(str(output_dir / "items"))
    spark.stop()


def train_tfrs_and_faiss(train: pd.DataFrame, movies: pd.DataFrame, output: Path, embedding_dim: int = 64, epochs: int = 3):
    """Train two-tower retrieval and save a FAISS item index plus IDs."""
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    import faiss
    user_ids = train.userId.astype(str).unique().tolist()
    item_ids = movies.movieId.astype(str).unique().tolist()
    interactions = tf.data.Dataset.from_tensor_slices({"user_id": train.userId.astype(str).values, "movie_id": train.movieId.astype(str).values}).shuffle(min(len(train), 100_000), seed=42).batch(4096)
    users = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
    items = tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None)
    class Tower(tfrs.Model):
        def __init__(self):
            super().__init__()
            self.user = tf.keras.Sequential([users, tf.keras.layers.Embedding(users.vocabulary_size(), embedding_dim)])
            self.item = tf.keras.Sequential([items, tf.keras.layers.Embedding(items.vocabulary_size(), embedding_dim)])
            candidates = tf.data.Dataset.from_tensor_slices(item_ids).batch(4096).map(self.item)
            self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=candidates))
        def compute_loss(self, features, training=False):
            return self.task(self.user(features["user_id"]), self.item(features["movie_id"]))
    model = Tower(); model.compile(optimizer=tf.keras.optimizers.Adagrad(.1)); model.fit(interactions, epochs=epochs, verbose=2)
    output.mkdir(parents=True, exist_ok=True); model.save_weights(str(output / "tfrs_tower.weights.h5"))
    vectors = model.item(tf.constant(item_ids)).numpy().astype("float32")
    faiss.normalize_L2(vectors); index = faiss.IndexFlatIP(embedding_dim); index.add(vectors); faiss.write_index(index, str(output / "items.faiss"))
    (output / "item_ids.json").write_text(json.dumps(item_ids), encoding="utf-8")


def train_ranker(train: pd.DataFrame, output: Path, epochs: int = 5):
    """A Wide & Deep-style binary ranker trained with train-only sampled negatives."""
    import tensorflow as tf
    positives = train[["userId", "movieId"]].drop_duplicates().copy(); positives["label"] = 1.0
    users = positives.userId.unique(); catalog = train.movieId.unique()
    rng = np.random.default_rng(42); known = set(zip(positives.userId, positives.movieId)); samples=[]
    for uid in users:
        candidates = [mid for mid in catalog if (uid, mid) not in known]
        if candidates: samples.extend((uid, mid, 0.0) for mid in rng.choice(candidates, min(4, len(candidates)), replace=False))
    neg = pd.DataFrame(samples, columns=["userId", "movieId", "label"])
    pairs = pd.concat([positives, neg], ignore_index=True)
    u_vocab = sorted(map(int, pairs.userId.unique())); i_vocab = sorted(map(int, pairs.movieId.unique()))
    u_in = tf.keras.Input(shape=(), dtype=tf.int32, name="userId"); i_in = tf.keras.Input(shape=(), dtype=tf.int32, name="movieId")
    u_idx = tf.keras.layers.StringLookup(vocabulary=list(map(str,u_vocab)), mask_token=None)(tf.strings.as_string(u_in))
    i_idx = tf.keras.layers.StringLookup(vocabulary=list(map(str,i_vocab)), mask_token=None)(tf.strings.as_string(i_in))
    u_emb = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(len(u_vocab)+1, 32)(u_idx)); i_emb = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(len(i_vocab)+1, 32)(i_idx))
    deep = tf.keras.layers.Concatenate()([u_emb, i_emb]); deep = tf.keras.layers.Dense(64, activation="relu")(deep); deep = tf.keras.layers.Dense(32, activation="relu")(deep)
    wide = tf.keras.layers.Concatenate()([tf.keras.layers.Embedding(len(u_vocab)+1, 1)(u_idx), tf.keras.layers.Embedding(len(i_vocab)+1, 1)(i_idx)])
    out = tf.keras.layers.Activation("sigmoid")(tf.keras.layers.Add()([tf.keras.layers.Dense(1)(deep), tf.keras.layers.Dense(1)(wide)]))
    model = tf.keras.Model({"userId":u_in,"movieId":i_in},out); model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
    model.fit({"userId":pairs.userId.astype("int32"),"movieId":pairs.movieId.astype("int32")},pairs.label,epochs=epochs,batch_size=4096,verbose=2)
    output.mkdir(parents=True,exist_ok=True); model.save(output / "wide_deep.keras")
    (output / "vocabularies.json").write_text(json.dumps({"users":u_vocab,"items":i_vocab}),encoding="utf-8")


def run(data_dir: Path, output: Path, k: int = 10, deep: bool = False, spark: bool = False):
    ratings_path = data_dir / "ratings.csv"
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating", "timestamp"])
    if ratings.empty or ratings[["userId", "movieId", "timestamp"]].isnull().any().any():
        raise ValueError("ratings.csv is empty or contains missing identifiers/timestamps")
    train, test = chronological_split(ratings)
    movies = pd.read_csv(data_dir / "movies.csv")
    metrics = compare_popularity_and_personalized(train, test, movies, k)
    output.mkdir(parents=True, exist_ok=True)
    metrics.update({"baseline": "global popularity excluding user's training history", "split": "per-user chronological 80/20", "ctr": None})
    (output / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    train.to_csv(output / "train_interactions.csv", index=False)
    test.to_csv(output / "test_interactions.csv", index=False)
    if spark:
        build_spark_features(ratings_path, output / "spark")
    if deep:
        train_tfrs_and_faiss(train, movies, output / "retrieval")
        train_ranker(train, output / "ranking")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--deep", action="store_true", help="Train TFRS retrieval, FAISS index, and Wide & Deep ranker")
    parser.add_argument("--spark", action="store_true", help="Build Spark Parquet feature tables")
    args = parser.parse_args()
    run(args.data, args.output, args.k, args.deep, args.spark)
