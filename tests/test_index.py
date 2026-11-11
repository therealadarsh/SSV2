from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ssv2.config import Settings
from ssv2.documents import Document, passage_text
from ssv2.embedding import l2_normalize
from ssv2.index import IndexManifestError, build_index
from ssv2.retrieval import DenseRetriever
from tests.fakes import FAKE_REVISION, HashEmbedder


def _docs() -> list[Document]:
    return [
        Document(
            id="vitamin",
            title="Vitamin D",
            text="Serum levels in winter.",
            source="fixture",
            license="MIT",
            metadata={"topic": "nutrition"},
        ),
        Document(
            id="graph",
            title="Graph networks",
            text="Message passing on citations.",
            source="fixture",
            license="MIT",
            metadata={"topic": "graphs"},
        ),
    ]


def test_search_ranks_the_matching_passage_first(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    documents = _docs()
    manifest = build_index(documents, embedder, tmp_path, dataset="fixture")
    assert manifest.document_count == 2
    assert manifest.embedding_model == embedder.model_id
    assert manifest.embedding_model_revision == FAKE_REVISION
    assert manifest.normalized is True
    assert manifest.metric == "inner_product"
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "docs.jsonl").is_file()
    assert (tmp_path / "index.faiss").is_file()

    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    retriever = DenseRetriever(tmp_path, embedder, settings)
    hits = retriever.search(passage_text(documents[1]), k=2)
    assert [hit.id for hit in hits] == ["graph", "vitamin"]
    assert hits[0].score > hits[1].score
    assert hits[0].metadata == {"topic": "graphs"}


def test_vectors_are_compared_by_direction_not_magnitude(tmp_path: Path) -> None:
    class ScaledEmbedder(HashEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
            vectors = super().embed_documents(texts)
            return vectors * np.array([[100.0], [0.01]], dtype=np.float32)

        def embed_query(self, text: str) -> np.ndarray:
            return super().embed_query(text) * np.float32(0.01)

    embedder = ScaledEmbedder()
    documents = _docs()
    build_index(documents, embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    hits = DenseRetriever(tmp_path, embedder, settings).search(passage_text(documents[0]), k=1)
    assert hits[0].id == "vitamin"
    # Cosine of a vector with itself is 1 regardless of the scale above.
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_filters_drop_nonmatching_documents_even_when_they_rank_higher(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    documents = _docs()
    build_index(documents, embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    retriever = DenseRetriever(tmp_path, embedder, settings)
    # The query is the graph passage, so "graph" would win unfiltered.
    hits = retriever.search(passage_text(documents[1]), k=2, filters={"topic": "nutrition"})
    assert [hit.id for hit in hits] == ["vitamin"]


def test_startup_refuses_an_index_built_with_a_different_model(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    build_index(_docs(), embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model="other/model",
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    with pytest.raises(IndexManifestError, match="other/model"):
        DenseRetriever(tmp_path, embedder, settings)


def test_refuses_a_revision_mismatch_and_a_duplicate_id(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    build_index(_docs(), embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision="b" * 40,
        index_dir=tmp_path,
    )
    with pytest.raises(IndexManifestError, match="revision"):
        DenseRetriever(tmp_path, embedder, settings)

    documents = _docs()
    documents[1] = documents[0].model_copy(update={"title": "Other"})
    with pytest.raises(ValueError, match="duplicate"):
        build_index(documents, embedder, tmp_path / "dup", dataset="fixture")


def test_missing_index_directory_is_an_error(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path / "missing",
    )
    with pytest.raises(FileNotFoundError):
        DenseRetriever(tmp_path / "missing", embedder, settings)


def test_normalize_helper_matches_index_scores() -> None:
    vector = np.array([3.0, 4.0], dtype=np.float32)
    assert np.isclose(float(np.dot(l2_normalize(vector), l2_normalize(vector))), 1.0)
