"""Dense results and BM25 results, merged by reciprocal rank fusion."""

from collections.abc import Mapping

from ssv2.bm25 import BM25Index
from ssv2.documents import passage_text
from ssv2.fusion import reciprocal_rank_fusion
from ssv2.retrieval import MIN_K, DenseRetriever, Hit, document_matches

# How many candidates each side contributes before fusion. Eval asks for 10.
DEFAULT_POOL = 100


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, *, pool: int = DEFAULT_POOL) -> None:
        if pool < MIN_K:
            raise ValueError(f"pool must be at least {MIN_K}")
        self.info = dense.info
        self._dense = dense
        self._pool = pool
        self._bm25 = BM25Index([passage_text(document) for document in dense.documents])

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        total = len(self._dense.documents)
        if total == 0 or k < MIN_K:
            return []
        pool = min(total, max(k, self._pool))
        dense_ids = [hit.id for hit in self._dense.search(query, pool, filters)]
        lexical_ids = self._lexical(query, pool, filters)
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
        by_id = {document.id: document for document in self._dense.documents}
        hits: list[Hit] = []
        for doc_id, score in fused[:k]:
            document = by_id[doc_id]
            hits.append(
                Hit(
                    id=document.id,
                    title=document.title,
                    text=document.text,
                    score=score,
                    metadata=dict(document.metadata),
                )
            )
        return hits

    def _lexical(self, query: str, pool: int, filters: Mapping[str, str] | None) -> list[str]:
        chosen: list[str] = []
        for index, _score in self._bm25.ranked(query):
            document = self._dense.documents[index]
            if filters and not document_matches(document, filters):
                continue
            chosen.append(document.id)
            if len(chosen) == pool:
                break
        return chosen
