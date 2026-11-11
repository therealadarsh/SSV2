import pytest

from ssv2.fusion import reciprocal_rank_fusion


def test_a_document_in_both_lists_outranks_either_alone() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert [doc_id for doc_id, _score in fused] == ["b", "a", "c"]
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_a_repeated_id_keeps_its_first_rank() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=0)
    assert fused[0] == ("a", 1.0)
    assert fused[1][0] == "b"


def test_rrf_rejects_a_negative_constant_and_accepts_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError, match="rrf k"):
        reciprocal_rank_fusion([["a"]], k=-1)
