"""recall@k, nDCG@k and MRR@k.

The definitions match the MTEB / TREC convention so a SciFact score can be compared with the
published number:

- gain is ``2^relevance - 1``
- the discount at 1-based rank ``r`` is ``log2(r + 1)``
- nDCG is DCG divided by the ideal DCG, and 0 when nothing is relevant
- recall@k is the fraction of relevant documents that appear in the top k
- MRR@k is ``1 / rank`` of the first relevant hit inside the cutoff, else 0

A repeated document id keeps its first rank and does not take another slot.
"""

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = _positives(relevant)
    if not positives:
        return 0.0
    found = set(_top(ranked_ids, k))
    return len(found & positives.keys()) / len(positives)


def mrr_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = _positives(relevant)
    for rank, doc_id in enumerate(_top(ranked_ids, k), start=1):
        if doc_id in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = _positives(relevant)
    if not positives:
        return 0.0
    gains = [positives.get(doc_id, 0) for doc_id in _top(ranked_ids, k)]
    ideal = sorted(positives.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no values to average")
    return sum(values) / len(values)


def _dcg(grades: Sequence[int]) -> float:
    total = 0.0
    for rank, grade in enumerate(grades, start=1):
        total += (2**grade - 1) / math.log2(rank + 1)
    return total


def _top(ranked_ids: Sequence[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    chosen: list[str] = []
    seen: set[str] = set()
    for doc_id in ranked_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        chosen.append(doc_id)
        if len(chosen) == k:
            break
    return chosen


def _positives(relevant: Mapping[str, int]) -> dict[str, int]:
    positives: dict[str, int] = {}
    for doc_id, grade in relevant.items():
        if grade < 0:
            raise ValueError(f"relevance grade for {doc_id} must be >= 0, got {grade}")
        if grade > 0:
            positives[doc_id] = grade
    return positives
