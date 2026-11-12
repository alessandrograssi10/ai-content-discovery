"""Retrieval metrics for the golden-set evaluation. Pure functions, no I/O.

  - R-Precision: precision computed at k=R, where R is the number of
    relevant items expected for THAT query (not a fixed k=5 for all
    queries). Standard IR metric for datasets with uneven relevant-item
    counts per query -- which this 21-title catalog has (R ranges from 1
    to 3 across the golden set). A fixed Precision@5 would artificially
    cap queries with R<5 (e.g. a query expecting only 1 relevant title
    can never score above 0.20 on a fixed top-5 window, even with a
    perfect top-1 result) -- R-Precision does not have this problem.

  - Recall@k: unaffected by the same issue (it can reach 1.0 regardless
    of R, as long as all relevant items appear somewhere in the top k).
"""


def r_precision(ranked_ids: list[str], expected_ids: set[str]) -> float:
    """Fraction of the expected items found in the top R results, R = len(expected_ids)."""
    r = len(expected_ids)
    if r == 0:
        return 0.0
    return len(expected_ids & set(ranked_ids[:r])) / r


def recall_at_k(ranked_ids: list[str], expected_ids: set[str], k: int = 5) -> float:
    """Fraction of the expected items found anywhere in the top k results."""
    if not expected_ids:
        return 0.0
    return len(expected_ids & set(ranked_ids[:k])) / len(expected_ids)
