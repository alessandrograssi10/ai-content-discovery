"""
Regression gate: the golden-set evaluation must not drop below eval/baseline.json.

Runs the real search API over the committed catalog, with each golden query's
embedding read from eval/query_embeddings.json instead of calling OpenAI.
A change to the search code, the enrichment prompt or the model (after
regenerating output/search_index.json) that lowers retrieval quality fails CI.

When a change improves the metrics, raise the baseline in the same PR.
"""

import json
from pathlib import Path

import pytest

from eval.metrics import r_precision, recall_at_k
from src.services.openai_client import EMBEDDING_MODEL

EVAL_DIR = Path(__file__).parent.parent / "eval"
TOP_K = 5


def load(name: str):
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_queries() -> list[dict]:
    return load("golden_queries.json")


@pytest.fixture(scope="module")
def query_embeddings() -> dict:
    return load("query_embeddings.json")


def test_query_embeddings_match_current_model(query_embeddings):
    assert query_embeddings["model"] == EMBEDDING_MODEL, (
        "eval/query_embeddings.json was built with a different embedding model. "
        "Regenerate it: python -m eval.build_query_embeddings"
    )


def test_every_golden_query_has_an_embedding(golden_queries, query_embeddings):
    missing = [g["query"] for g in golden_queries if g["query"] not in query_embeddings["embeddings"]]
    assert not missing, (
        f"Missing embeddings for {missing}. Regenerate: python -m eval.build_query_embeddings"
    )


def test_golden_set_does_not_regress(api_client, fake_llm, golden_queries, query_embeddings):
    baseline = load("baseline.json")
    per_query = []

    for entry in golden_queries:
        fake_llm.next_embedding = query_embeddings["embeddings"][entry["query"]]
        response = api_client.get("/search", params={"q": entry["query"], "top_k": TOP_K})
        assert response.status_code == 200
        ranked_ids = [r["content_id"] for r in response.json()["results"]]
        expected = set(entry["expected_content_ids"])
        per_query.append(
            (entry["query"], r_precision(ranked_ids, expected), recall_at_k(ranked_ids, expected))
        )

    avg_r_precision = sum(rp for _, rp, _ in per_query) / len(per_query)
    avg_recall = sum(rc for _, _, rc in per_query) / len(per_query)

    report = "\n".join(f"  R-Prec {rp:.2f}  Recall@5 {rc:.2f}  {q}" for q, rp, rc in per_query)
    summary = (
        f"Avg R-Precision {avg_r_precision:.4f} (baseline {baseline['r_precision']}), "
        f"Avg Recall@5 {avg_recall:.4f} (baseline {baseline['recall_at_5']})\n{report}"
    )
    assert avg_r_precision >= baseline["r_precision"], summary
    assert avg_recall >= baseline["recall_at_5"], summary
