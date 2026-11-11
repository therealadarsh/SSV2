"""A second search when the first hit is weak, merged with the first by RRF.

This is the agentic loop with the model removed: the only action is search, and the
policy is fixed so the SciFact comparison does not depend on an API key. A strong first
hit is returned as-is, which is one search, the same as the dense retriever.
"""

from collections.abc import Mapping

from ssv2.fusion import reciprocal_rank_fusion
from ssv2.retrieval import Hit, Retriever

# Dense cosine. A first hit at or above this is kept and no second search runs.
# 0.55 never fires on SciFact (every top hit is stronger). 0.70 makes nDCG worse.
# 0.80 is the cutoff that beat dense. See DECISIONS.md.
FOLLOWUP_BELOW = 0.8


class AgenticRetriever:
    def __init__(self, base: Retriever, *, followup_below: float = FOLLOWUP_BELOW) -> None:
        self.info = base.info
        self._base = base
        self._followup_below = followup_below
        self.searches = 0

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        self.searches += 1
        first = self._base.search(query, k, filters)
        if not first or first[0].score >= self._followup_below:
            return first
        self.searches += 1
        second = self._base.search(f"{query} {first[0].title}", k, filters)
        fused = reciprocal_rank_fusion([[hit.id for hit in first], [hit.id for hit in second]])
        by_id = {hit.id: hit for hit in [*first, *second]}
        return [by_id[doc_id].model_copy(update={"score": score}) for doc_id, score in fused[:k]]
