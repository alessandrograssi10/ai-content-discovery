"""Search API tests: the real app and index, with OpenAI replaced by a fake client."""

import numpy as np

from src.search_api import SIMILARITY_THRESHOLD


def embedding_of(index_items, title):
    return next(item["embedding"] for item in index_items if item["title"] == title)


def test_health_reports_catalog_size(api_client, index_items):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "catalog_size": len(index_items)}


def test_query_matching_a_title_ranks_it_first(api_client, fake_llm, index_items):
    fake_llm.next_embedding = embedding_of(index_items, "Inception")
    response = api_client.get("/search", params={"q": "dream heist"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "dream heist"
    assert body["results"][0]["title"] == "Inception"
    assert body["results"][0]["score"] > 0.999
    assert fake_llm.queries == ["dream heist"]


def test_results_are_sorted_and_above_threshold(api_client, fake_llm, index_items):
    fake_llm.next_embedding = embedding_of(index_items, "Breaking Bad")
    results = api_client.get("/search", params={"q": "x", "top_k": 20}).json()["results"]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(score >= SIMILARITY_THRESHOLD for score in scores)


def test_top_k_limits_results(api_client, fake_llm, index_items):
    fake_llm.next_embedding = embedding_of(index_items, "Inception")
    results = api_client.get("/search", params={"q": "x", "top_k": 2}).json()["results"]
    assert len(results) == 2


def test_unrelated_query_returns_empty_list(api_client, fake_llm, index_items):
    # A random direction in 1536-d space is near-orthogonal to every title.
    dim = len(index_items[0]["embedding"])
    fake_llm.next_embedding = np.random.default_rng(0).standard_normal(dim).tolist()
    response = api_client.get("/search", params={"q": "quantum tax law"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_whitespace_query_is_rejected(api_client, fake_llm):
    response = api_client.get("/search", params={"q": "   "})
    assert response.status_code == 400
    assert fake_llm.queries == []


def test_empty_query_is_rejected(api_client):
    assert api_client.get("/search", params={"q": ""}).status_code == 422


def test_missing_query_is_rejected(api_client):
    assert api_client.get("/search").status_code == 422


def test_top_k_out_of_range_is_rejected(api_client):
    assert api_client.get("/search", params={"q": "x", "top_k": 0}).status_code == 422
    assert api_client.get("/search", params={"q": "x", "top_k": 21}).status_code == 422
