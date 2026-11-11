"""Cross-encoder reranking of the top of another retriever's list.

The cross-encoder is loaded only when a reranking retriever is constructed, so the
default dense path does not download it.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Protocol

from ssv2.config import Settings
from ssv2.embedding import prepare_hf_home
from ssv2.retrieval import Hit, Retriever

RERANK_DEPTH = 30


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


class RerankedRetriever:
    def __init__(self, base: Retriever, reranker: Reranker, *, depth: int = RERANK_DEPTH) -> None:
        if depth < 1:
            raise ValueError("rerank depth must be at least 1")
        self.info = base.info
        self._base = base
        self._reranker = reranker
        self._depth = depth

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        if k < 1:
            return []
        hits = self._base.search(query, max(k, self._depth), filters)
        head = hits[: self._depth]
        tail = hits[self._depth :]
        if not head:
            return []
        passages = [_passage(hit) for hit in head]
        scores = [float(score) for score in self._reranker.score(query, passages)]
        if len(scores) != len(head):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(head)} passages")
        order = sorted(range(len(head)), key=lambda index: (-scores[index], index))
        reranked = [head[index].model_copy(update={"score": scores[index]}) for index in order]
        return [*reranked, *tail][:k]


class CrossEncoderReranker:
    """``cross-encoder/ms-marco-MiniLM-L6-v2`` at the revision in settings."""

    def __init__(self, settings: Settings) -> None:
        prepare_hf_home()
        from sentence_transformers import CrossEncoder

        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        self._model = CrossEncoder(
            settings.rerank_model,
            revision=settings.rerank_model_revision,
            device="cpu",
            local_files_only=offline,
            trust_remote_code=False,
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        predicted = self._model.predict(
            [(query, passage) for passage in passages],
            show_progress_bar=False,
        )
        return [float(value) for value in predicted]


def _passage(hit: Hit) -> str:
    title = hit.title.strip()
    text = hit.text.strip()
    if title and text:
        return f"{title} {text}"
    return title or text
