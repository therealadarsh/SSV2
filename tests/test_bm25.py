import math

import pytest

from ssv2.bm25 import BM25Index


def test_okapi_score_matches_the_hand_worked_value() -> None:
    index = BM25Index(["alpha alpha beta", "beta gamma"])
    ranked = index.ranked("alpha beta")
    assert [item[0] for item in ranked] == [0, 1]
    idf_alpha = math.log(2)
    idf_beta = math.log(1.2)
    norm = 1.5 * (0.25 + 0.75 * 3 / 2.5)
    expected = idf_alpha * 5 / (2 + norm) + idf_beta * 2.5 / (1 + norm)
    assert ranked[0][1] == pytest.approx(expected)
    assert ranked[0][1] == pytest.approx(1.0976664583667337)


def test_bm25_boundaries() -> None:
    assert BM25Index(["alpha"]).ranked("   ") == []
    assert BM25Index([]).ranked("alpha") == []
    # A query term that is in neither document contributes nothing, and the hit is omitted.
    assert BM25Index(["alpha"]).ranked("missing") == []
    with pytest.raises(ValueError, match="k1"):
        BM25Index(["alpha"], k1=-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        BM25Index(["alpha"], b=1.1)
