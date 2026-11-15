import pandas as pd
from src.pipeline import chronological_split, ranking_metrics, compare_popularity_and_personalized


def sample():
    return pd.DataFrame({"userId": [1, 1, 1, 2, 2, 2], "movieId": [1, 2, 3, 1, 3, 4], "rating": [5] * 6, "timestamp": [1, 2, 3, 1, 2, 3]})


def test_split_is_chronological_and_disjoint():
    train, test = chronological_split(sample(), 2 / 3)
    assert set(train.index).isdisjoint(test.index)
    assert train.groupby("userId").timestamp.max().lt(test.groupby("userId").timestamp.min()).all()


def test_metrics_are_bounded():
    train, test = chronological_split(sample(), 2 / 3)
    result = ranking_metrics(test, train, 2)
    assert 0 <= result["recall_at_k"] <= 1
    assert 0 <= result["ndcg_at_k"] <= 1


def test_sparse_users_keep_train_and_holdout_when_possible():
    data = pd.DataFrame({"userId":[1,1,2],"movieId":[10,11,12],"timestamp":[1,2,1]})
    train, test = chronological_split(data)
    assert set(train.loc[train.userId == 1, "movieId"]) == {10}
    assert set(test.loc[test.userId == 1, "movieId"]) == {11}
    assert 2 in set(train.userId)
    assert 2 not in set(test.userId)


def test_comparison_uses_shared_catalog_and_returns_no_fake_ctr():
    train, test = chronological_split(sample(), 2 / 3)
    movies = pd.DataFrame({"movieId":[1,2,3,4],"genres":["Drama","Comedy","Drama|Romance","Action"]})
    results = compare_popularity_and_personalized(train, test, movies, 2)
    assert set(results) == {"popularity", "genre_personalized"}
    assert all(x["ctr"] is None for x in results.values())
    assert all(0 <= x["ndcg_at_k"] <= 1 for x in results.values())


def test_invalid_cutoff_rejected():
    import pytest
    with pytest.raises(ValueError): chronological_split(sample(), 1.0)
