"""
Evaluation script for the semantic search endpoint.

Requires the search server to be running:
    uvicorn src.search_api:app --reload

Run with:
    python -m eval.run_eval

Metrics:

  - R-Precision: precision computed at k=R, where R is the number of
    relevant items expected for THAT query (not a fixed k=5 for all
    queries). Standard IR metric for datasets with uneven relevant-item
    counts per query -- which this 21-title catalog has (R ranges from 1
    to 3 across the golden set). A fixed Precision@5 would artificially
    cap queries with R<5 (e.g. a query expecting only 1 relevant title
    can never score above 0.20 on a fixed top-5 window, even with a
    perfect top-1 result) -- R-Precision does not have this problem.

  - Recall@5: unaffected by the same issue (it can reach 1.0 regardless
    of R, as long as all relevant items appear somewhere in the top 5),
    so it is kept as-is rather than replaced.

"""

import json
from pathlib import Path

import httpx

SEARCH_URL = "http://localhost:8000/search"
TOP_K = 5
GOLDEN_PATH = Path(__file__).parent / "golden_queries.json"


def run_eval() -> None:
    golden_queries = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    try:
        httpx.get("http://localhost:8000/health", timeout=3).raise_for_status()
    except Exception:
        print("Server non raggiungibile — avvia prima 'uvicorn src.search_api:app' in un altro terminale.")
        return

    r_precision_scores: list[float] = []
    recall_scores: list[float] = []

    for entry in golden_queries:
        query = entry["query"]
        expected_ids = set(entry["expected_content_ids"])
        R = len(expected_ids)

        response = httpx.get(SEARCH_URL, params={"q": query, "top_k": TOP_K}, timeout=15)
        response.raise_for_status()
        results = response.json()["results"]

        # Recall@5: computed over the full top-5 window, as before.
        returned_ids_top5 = {r["content_id"] for r in results}
        hits_top5 = expected_ids & returned_ids_top5
        recall = len(hits_top5) / R if R > 0 else 0.0

        # R-Precision: computed over only the top-R results, R = number
        # of relevant items expected for THIS query, not a fixed 5.
        top_r_results = results[:R]
        returned_ids_top_r = {r["content_id"] for r in top_r_results}
        hits_top_r = expected_ids & returned_ids_top_r
        r_precision = len(hits_top_r) / R if R > 0 else 0.0

        r_precision_scores.append(r_precision)
        recall_scores.append(recall)

        print(f"Query      : {query}")
        print(f"Expected   : {sorted(expected_ids)}  (R={R})")
        got = [f"{r['content_id']} ({r['title']}, score={r['score']})" for r in results]
        print(f"Got (top-5): {got}")
        print(f"Hits (top-5, for recall)  : {sorted(hits_top5) or 'none'}")
        print(f"Hits (top-{R}, for R-precision): {sorted(hits_top_r) or 'none'}")
        print(f"R-Precision: {r_precision:.2f}  |  Recall@5: {recall:.2f}")
        print()

    avg_r_precision = sum(r_precision_scores) / len(r_precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    print("=" * 60)
    print(f"Avg R-Precision: {avg_r_precision:.2f}")
    print(f"Avg Recall@5   : {avg_recall:.2f}")
    print()
    print(
        "Note: R-Precision uses k = number of relevant items expected for "
        "each query (R), instead of a fixed k=5 -- standard IR practice for "
        "datasets with uneven relevant-item counts per query, which this "
        "21-title catalog has (R ranges from 1 to 3 across the golden set). "
        "A fixed Precision@5 would unfairly cap queries with R<5 even when "
        "the top result is a perfect match; R-Precision does not have this "
        "problem, though it can penalize recall if a correct match ranks "
        "just beyond position R."
    )


if __name__ == "__main__":
    run_eval()