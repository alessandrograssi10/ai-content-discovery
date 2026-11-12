import pytest

from eval.metrics import r_precision, recall_at_k


def test_r_precision_perfect_top_one():
    # R=1: a correct top-1 result scores 1.0 (a fixed Precision@5 would cap it at 0.2).
    assert r_precision(["1002", "1010", "1021"], {"1002"}) == 1.0


def test_r_precision_only_looks_at_top_r():
    # R=2: the second expected item is at rank 3, outside the top-R window.
    assert r_precision(["1013", "1007", "1019"], {"1013", "1019"}) == 0.5


def test_recall_counts_hits_anywhere_in_top_k():
    assert recall_at_k(["1013", "1007", "1019"], {"1013", "1019"}, k=5) == 1.0


def test_recall_ignores_hits_beyond_k():
    ranked = ["a", "b", "c", "d", "e", "1003"]
    assert recall_at_k(ranked, {"1003"}, k=5) == 0.0


def test_no_results_scores_zero():
    assert r_precision([], {"1003"}) == 0.0
    assert recall_at_k([], {"1003"}) == 0.0


@pytest.mark.parametrize("metric", [r_precision, recall_at_k])
def test_empty_expected_set_scores_zero(metric):
    assert metric(["1001"], set()) == 0.0
