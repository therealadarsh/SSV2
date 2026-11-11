import pytest

from ssv2.latency import percentile


def test_nearest_rank_percentiles() -> None:
    samples = [10.0, 20.0, 30.0, 40.0, 100.0]
    # ceil(0.95 * 5) = 5, so p95 is the largest sample.
    assert percentile(samples, 95) == 100.0
    # ceil(0.50 * 5) = 3, the middle sample.
    assert percentile(samples, 50) == 30.0
    assert percentile(samples, 100) == 100.0
    assert percentile(samples, 0) == 10.0


def test_percentile_sorts_and_accepts_one_sample() -> None:
    assert percentile([5.0, 1.0, 3.0], 100) == 5.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_empty_samples_and_out_of_range() -> None:
    with pytest.raises(ValueError, match="no samples"):
        percentile([], 95)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], -0.1)
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1.0], 100.1)
