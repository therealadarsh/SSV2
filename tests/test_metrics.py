"""Hand-worked metric cases. Expected numbers are calculated by hand, not by the code under test."""

import math

import pytest

from ssv2.evals.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k

_LOG3 = math.log2(3)


def test_perfect_hit_at_rank_one() -> None:
    ranked = ["a"]
    grades = {"a": 1}
    assert recall_at_k(ranked, grades, 1) == 1.0
    assert mrr_at_k(ranked, grades, 1) == 1.0
    assert ndcg_at_k(ranked, grades, 1) == 1.0


def test_relevant_document_at_rank_two() -> None:
    ranked = ["miss", "hit"]
    grades = {"hit": 1}
    assert recall_at_k(ranked, grades, 2) == 1.0
    assert mrr_at_k(ranked, grades, 2) == 0.5
    # DCG = 1/log2(3), ideal DCG = 1.
    assert ndcg_at_k(ranked, grades, 2) == pytest.approx(1 / _LOG3)


def test_cutoff_hides_the_only_relevant_document() -> None:
    ranked = ["a", "b", "c"]
    grades = {"c": 1}
    assert recall_at_k(ranked, grades, 2) == 0.0
    assert mrr_at_k(ranked, grades, 2) == 0.0
    assert ndcg_at_k(ranked, grades, 2) == 0.0


def test_no_relevant_documents_scores_zero() -> None:
    assert recall_at_k(["a"], {}, 1) == 0.0
    assert mrr_at_k(["a"], {"a": 0}, 1) == 0.0
    assert ndcg_at_k(["a"], {"a": 0}, 5) == 0.0


def test_graded_relevance_prefers_the_higher_grade_first() -> None:
    ranked = ["low", "high"]
    grades = {"low": 1, "high": 3}
    # DCG = 1/log2(2) + (2^3 - 1)/log2(3); ideal puts grade 3 first.
    dcg = 1 + 7 / _LOG3
    ideal = 7 + 1 / _LOG3
    assert ndcg_at_k(ranked, grades, 2) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["high", "low"], grades, 2) == 1.0


def test_duplicate_id_keeps_its_first_rank() -> None:
    assert ndcg_at_k(["a", "a"], {"a": 1}, 2) == 1.0
    assert recall_at_k(["a", "a", "b"], {"a": 1, "b": 1}, 2) == 1.0
    assert mrr_at_k(["miss", "hit", "hit"], {"hit": 1}, 3) == 0.5


def test_k_past_the_end_of_the_ranking() -> None:
    ranked = ["a"]
    grades = {"a": 1, "b": 1}
    assert recall_at_k(ranked, grades, 10) == 0.5
    assert mrr_at_k(ranked, grades, 10) == 1.0
    assert ndcg_at_k(ranked, grades, 10) == pytest.approx(1 / (1 + 1 / _LOG3))


def test_k_below_one_and_a_negative_grade_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        recall_at_k(["a"], {"a": 1}, 0)
    with pytest.raises(ValueError, match=">= 0"):
        ndcg_at_k(["a"], {"a": -1}, 1)


def test_mean_rejects_an_empty_list() -> None:
    assert mean([1.0, 0.0]) == 0.5
    with pytest.raises(ValueError, match="no values"):
        mean([])
