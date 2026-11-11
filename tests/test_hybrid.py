import datetime
from collections.abc import Sequence
from pathlib import Path

import pytest

from ssv2.config import Settings
from ssv2.documents import Document, passage_text
from ssv2.fusion import reciprocal_rank_fusion
from ssv2.hybrid import HybridRetriever
from ssv2.index import build_index
from ssv2.rerank import RerankedRetriever
from ssv2.retrieval import DenseRetriever
from tests.fakes import HashEmbedder


def _settings(index_dir: Path, embedder: HashEmbedder) -> Settings:
    return Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=index_dir,
    )


def _corpus() -> list[Document]:
    return [
        Document(
            id="old",
            title="Old apples",
            text="apples in the archive",
            source="notes",
            license="MIT",
            date=datetime.date(2020, 1, 1),
            metadata={"category": "cs.CL", "categories": "cs.CL", "date": "2020-01-01"},
        ),
        Document(
            id="new",
            title="New bridges",
            text="bridges and rivers",
            source="notes",
            license="MIT",
            date=datetime.date(2024, 6, 1),
            metadata={"category": "cs.AI", "categories": "cs.AI math.ST", "date": "2024-06-01"},
        ),
        Document(
            id="undated",
            title="Undated",
            text="no publication day",
            source="notes",
            license="MIT",
            metadata={"category": "cs.CL", "categories": "cs.CL"},
        ),
    ]


def test_hybrid_score_is_the_fusion_of_the_two_rankings(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    documents = _corpus()
    build_index(documents, embedder, tmp_path, dataset="notes")
    dense = DenseRetriever(tmp_path, embedder, _settings(tmp_path, embedder))
    hybrid = HybridRetriever(dense, pool=10)
    query = passage_text(documents[1])
    hits = hybrid.search(query, k=2)
    dense_ids = [hit.id for hit in dense.search(query, k=2)]
    lexical_ids = hybrid._lexical(query, pool=2, filters=None)
    fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
    assert [(hit.id, hit.score) for hit in hits] == fused


def test_hybrid_date_and_category_filters(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    build_index(_corpus(), embedder, tmp_path, dataset="notes")
    hybrid = HybridRetriever(
        DenseRetriever(tmp_path, embedder, _settings(tmp_path, embedder)), pool=10
    )
    dated = hybrid.search("bridges", k=5, filters={"date_from": "2024-01-01"})
    assert [hit.id for hit in dated] == ["new"]
    # cs.CL is not the primary category of the second paper, and it is not in its token list.
    only_cl = hybrid.search("apples", k=5, filters={"category": "cs.CL"})
    assert {hit.id for hit in only_cl} == {"old", "undated"}
    # math.ST is a secondary category, so the category filter still matches.
    secondary = hybrid.search("bridges", k=5, filters={"category": "math.ST"})
    assert [hit.id for hit in secondary] == ["new"]
    with pytest.raises(ValueError, match="date_from"):
        hybrid.search("bridges", k=5, filters={"date_from": "June"})
    # A document with no date cannot satisfy a date bound, even when the bound is wide.
    query = "Undated no publication day"
    assert any(hit.id == "undated" for hit in hybrid.search(query, k=5))
    filtered = hybrid.search(query, k=5, filters={"date_to": "2030-01-01"})
    assert all(hit.id != "undated" for hit in filtered)


def test_reranker_moves_the_preferred_passage_first(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    documents = _corpus()
    build_index(documents, embedder, tmp_path, dataset="notes")
    dense = DenseRetriever(tmp_path, embedder, _settings(tmp_path, embedder))

    class PreferOld:
        def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
            del query
            return [1.0 if "apples" in passage else 0.0 for passage in passages]

    query = passage_text(documents[1])
    assert dense.search(query, k=1)[0].id == "new"
    reranked = RerankedRetriever(dense, PreferOld(), depth=2).search(query, k=1)
    assert reranked[0].id == "old"
    assert reranked[0].score == 1.0


def test_reranker_rejects_a_short_score_list_and_a_zero_depth(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    build_index(_corpus(), embedder, tmp_path, dataset="notes")
    dense = DenseRetriever(tmp_path, embedder, _settings(tmp_path, embedder))

    class Short:
        def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
            del query, passages
            return []

    with pytest.raises(ValueError, match="scores"):
        RerankedRetriever(dense, Short(), depth=1).search("bridges", k=1)
    with pytest.raises(ValueError, match="depth"):
        RerankedRetriever(dense, Short(), depth=0)
