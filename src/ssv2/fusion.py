"""Reciprocal rank fusion. Rank is 1-based and a repeated id keeps its first rank."""

from collections.abc import Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = RRF_K
) -> list[tuple[str, float]]:
    if k < 0:
        raise ValueError("rrf k must be >= 0")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return [(doc_id, scores[doc_id]) for doc_id in ordered]
