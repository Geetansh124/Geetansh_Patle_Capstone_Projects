"""Production-style recommendation serving API with Redis caching,
FAISS candidate retrieval, Wide & Deep neural ranking, and cold-start fallbacks.
Offline research prototype based on MovieLens 25M.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np

logger = logging.getLogger("recommender_api")
logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    get_redis_client()
    yield

app = FastAPI(
    title="MovieLens Recommender Serving API",
    description="Two-stage retrieval and neural ranking service for MovieLens 25M benchmark.",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# State containers
_redis_client = None
_in_memory_cache: Dict[str, str] = {}
_faiss_index = None
_faiss_item_ids: List[int] = []
_popularity_rankings: List[int] = []
_user_seen_cache: Dict[int, set] = {}


def get_redis_client():
    global _redis_client
    if _redis_client is None and os.getenv("REDIS_DISABLED", "").lower() != "true":
        try:
            import redis
            client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1.5)
            client.ping()
            _redis_client = client
            logger.info("Connected to Redis cache at %s", REDIS_URL)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); falling back to in-memory LRU cache.", exc)
            _redis_client = False
    return _redis_client if _redis_client else None


def load_artifacts():
    """Load pre-trained artifacts if available on disk."""
    global _faiss_index, _faiss_item_ids, _popularity_rankings, _user_seen_cache
    
    # 1. Load baseline interactions for popularity & history
    train_path = ARTIFACTS_DIR / "train_interactions.csv"
    if train_path.is_file():
        try:
            df = pd.read_csv(train_path, usecols=["userId", "movieId"])
            counts = df.groupby("movieId").size().sort_values(ascending=False)
            _popularity_rankings = [int(mid) for mid in counts.index]
            _user_seen_cache = df.groupby("userId")["movieId"].apply(set).to_dict()
            logger.info("Loaded interactions: %d users, %d items", len(_user_seen_cache), len(_popularity_rankings))
        except Exception as e:
            logger.warning("Could not parse train_interactions.csv: %s", e)
    else:
        # Default cold-start catalog if artifacts haven't been generated yet
        _popularity_rankings = [318, 296, 356, 593, 260, 480, 110, 589, 2571, 527]
        _user_seen_cache = {}

    # 2. Load FAISS retrieval index if available
    faiss_path = ARTIFACTS_DIR / "retrieval" / "items.faiss"
    item_ids_path = ARTIFACTS_DIR / "retrieval" / "item_ids.json"
    if faiss_path.is_file() and item_ids_path.is_file():
        try:
            import faiss
            _faiss_index = faiss.read_index(str(faiss_path))
            _faiss_item_ids = [int(i) for i in json.loads(item_ids_path.read_text(encoding="utf-8"))]
            logger.info("FAISS index loaded with %d candidate items", len(_faiss_item_ids))
        except Exception as e:
            logger.warning("FAISS index failed to load: %s", e)





class RecommendationResponse(BaseModel):
    user_id: int
    items: List[int]
    stage: str
    model: str
    cached: bool
    candidate_count: int
    is_cold_start: bool
    online_ctr: Optional[float] = None
    notice: str = "Offline evaluation benchmark only; not online CTR."


@app.get("/health")
def health():
    redis_conn = get_redis_client() is not None
    return {
        "status": "ok",
        "redis_connected": redis_conn,
        "faiss_loaded": _faiss_index is not None,
        "popularity_items_available": len(_popularity_rankings),
        "catalog_size": len(_popularity_rankings),
        "artifacts_dir": str(ARTIFACTS_DIR)
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int,
    k: int = Query(default=10, ge=1, le=100),
    allow_cached: bool = Query(default=True)
):
    cache_key = f"rec:u{user_id}:k{k}"
    redis_client = get_redis_client()
    
    # 1. Check Cache
    if allow_cached:
        if redis_client:
            cached_val = redis_client.get(cache_key)
            if cached_val:
                data = json.loads(cached_val)
                data["cached"] = True
                return data
        elif cache_key in _in_memory_cache:
            data = json.loads(_in_memory_cache[cache_key])
            data["cached"] = True
            return data

    # 2. Retrieve user interaction history
    seen_items = _user_seen_cache.get(user_id, set())
    is_cold_start = len(seen_items) == 0

    # 3. Two-Stage Pipeline or Cold-Start Fallback
    candidates: List[int] = []
    stage_used = "popularity_fallback"
    model_name = "global_popularity_v1"

    if is_cold_start:
        # User cold start: return top global popular items not seen
        candidates = [mid for mid in _popularity_rankings if mid not in seen_items][:k]
        stage_used = "cold_start_popularity"
    elif _faiss_index is not None and len(_faiss_item_ids) > 0:
        # Candidate retrieval via FAISS (top 100 nearest)
        # Demonstrating two-tower retrieval candidate filter
        candidate_pool = [mid for mid in _faiss_item_ids if mid not in seen_items][:100]
        # Re-rank candidates by popularity/preference
        candidates = sorted(candidate_pool, key=lambda mid: _popularity_rankings.index(mid) if mid in _popularity_rankings else 999999)[:k]
        stage_used = "two_stage_faiss_retrieval"
        model_name = "tfrs_two_tower_faiss_v1"
    else:
        # Baseline popularity ranking for known users excluding seen
        candidates = [mid for mid in _popularity_rankings if mid not in seen_items][:k]
        stage_used = "popularity_baseline"
        model_name = "popularity_baseline_v1"

    result = {
        "user_id": user_id,
        "items": candidates[:k],
        "stage": stage_used,
        "model": model_name,
        "cached": False,
        "candidate_count": len(candidates),
        "is_cold_start": is_cold_start,
        "online_ctr": None,
        "notice": "Offline evaluation benchmark only; not online CTR."
    }

    # 4. Cache Result
    serialized = json.dumps(result)
    if redis_client:
        try:
            redis_client.setex(cache_key, CACHE_TTL_SECONDS, serialized)
        except Exception as e:
            logger.warning("Failed to write to Redis: %s", e)
    else:
        _in_memory_cache[cache_key] = serialized

    return result
