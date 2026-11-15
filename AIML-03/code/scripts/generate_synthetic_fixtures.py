"""Generate synthetic MovieLens fixtures and run baseline pipeline for local evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.pipeline import run


def generate_synthetic_movielens(out_dir: Path, num_users: int = 50, num_movies: int = 40):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    genres_pool = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Documentary"]
    movies_records = []
    for mid in range(1, num_movies + 1):
        g = np.random.choice(genres_pool, size=np.random.randint(1, 3), replace=False)
        movies_records.append({
            "movieId": mid,
            "title": f"Synthetic Film {mid} (2026)",
            "genres": "|".join(g)
        })
    movies_df = pd.DataFrame(movies_records)
    movies_df.to_csv(out_dir / "movies.csv", index=False)
    
    # Generate ratings
    ratings_records = []
    base_timestamp = 1600000000
    for uid in range(1, num_users + 1):
        interacted_count = np.random.randint(5, 25)
        mids = np.random.choice(range(1, num_movies + 1), size=interacted_count, replace=False)
        for step, mid in enumerate(mids):
            ratings_records.append({
                "userId": uid,
                "movieId": mid,
                "rating": round(np.random.choice([3.0, 3.5, 4.0, 4.5, 5.0]), 1),
                "timestamp": base_timestamp + step * 3600 + np.random.randint(1, 600)
            })
    ratings_df = pd.DataFrame(ratings_records)
    ratings_df.to_csv(out_dir / "ratings.csv", index=False)
    print(f"Generated {len(ratings_df)} synthetic ratings across {num_users} users and {num_movies} movies in {out_dir}")
    return out_dir


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent.parent
    data_dir = base / "data" / "synthetic_ml"
    generate_synthetic_movielens(data_dir)
    artifacts_dir = base / "artifacts"
    print("Running baseline chronological pipeline on synthetic dataset...")
    run(data_dir, artifacts_dir, k=10, deep=False, spark=False)
    print(f"Baseline artifacts saved to {artifacts_dir}")
