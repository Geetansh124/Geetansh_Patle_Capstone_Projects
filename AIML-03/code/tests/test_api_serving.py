import importlib.util
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import pandas as pd

api_path = Path(__file__).resolve().parent.parent / "src" / "api.py"
spec = importlib.util.spec_from_file_location("aiml03_api", api_path)
api_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_mod)

app = api_mod.app
load_artifacts = api_mod.load_artifacts
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_mock_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setenv("REDIS_DISABLED", "true")
    monkeypatch.setattr(api_mod, "_redis_client", None)
    monkeypatch.setattr(api_mod, "_in_memory_cache", {})
    
    # Create test interactions
    train_csv = tmp_path / "train_interactions.csv"
    train_df = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3],
        "movieId": [10, 20, 30, 10, 40, 50]
    })
    train_df.to_csv(train_csv, index=False)
    load_artifacts()


def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["catalog_size"] > 0


def test_recommend_existing_user():
    res = client.get("/recommend/1?k=2&allow_cached=false")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == 1
    assert len(data["items"]) <= 2
    # Ensure already seen items (10, 20, 30) are filtered out
    assert 10 not in data["items"]
    assert 20 not in data["items"]
    assert 30 not in data["items"]
    assert data["is_cold_start"] is False
    assert data["online_ctr"] is None


def test_recommend_cold_start_user():
    res = client.get("/recommend/99999?k=3&allow_cached=false")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == 99999
    assert data["is_cold_start"] is True
    assert len(data["items"]) == 3
    assert data["items"][0] == 10


def test_caching_behavior():
    res1 = client.get("/recommend/2?k=2&allow_cached=true")
    assert res1.status_code == 200
    assert res1.json()["cached"] is False
    
    res2 = client.get("/recommend/2?k=2&allow_cached=true")
    assert res2.status_code == 200
    assert res2.json()["cached"] is True
    assert res2.json()["items"] == res1.json()["items"]


def test_invalid_k_rejected():
    res_zero = client.get("/recommend/1?k=0")
    assert res_zero.status_code == 422
    
    res_huge = client.get("/recommend/1?k=1000")
    assert res_huge.status_code == 422
