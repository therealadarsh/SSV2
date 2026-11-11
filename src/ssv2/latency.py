"""Nearest-rank percentiles for latency samples.

Rank is 1-based: ``ceil(pct / 100 * n)``. p95 of five samples is the largest one,
because ``ceil(4.75)`` is 5. p0 is the smallest sample.
"""

import math
from collections.abc import Sequence


def percentile(samples: Sequence[float], pct: float) -> float:
    if not samples:
        raise ValueError("no samples")
    if pct < 0 or pct > 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(samples)
    if pct == 0:
        return float(ordered[0])
    rank = math.ceil(pct / 100 * len(ordered))
    return float(ordered[rank - 1])
