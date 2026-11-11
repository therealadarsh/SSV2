"""Embedders and retrievers that stand in for the real model in unit tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np

from ssv2.documents import Document
from ssv2.index import IndexManifest
from ssv2.retrieval import Hit

FAKE_REVISION = "a" * 40


class HashEmbedder:
    """Deterministic vectors from the text, with no network and no model weights.

    The same string always maps to the same vector, including across processes
    (unlike Python's salted `hash`).
    """

    def __init__(self, dimension: int = 8) -> None:
        self.model_id = "fake/hash"
        self.revision = FAKE_REVISION
        self.dimension = dimension
        # The fake does not prepend the BGE instruction; query text is embedded as given.
        self.query_instruction = ""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._vector(text) for text in texts]
        if not rows:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        return np.random.default_rng(seed).standard_normal(self.dimension).astype(np.float32)


class StubRetriever:
    """Returns one fixed hit and records the call. For HTTP tests that are not about ranking."""

    def __init__(self) -> None:
        self.info = IndexManifest(
            dataset="stub",
            document_count=1,
            embedding_model="fake/hash",
            embedding_model_revision=FAKE_REVISION,
            embedding_dimension=8,
            built_at=datetime(2026, 11, 11, tzinfo=UTC),
            query_instruction="",
        )
        self.calls: list[tuple[str, int, Mapping[str, str] | None]] = []

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        self.calls.append((query, k, filters))
        return [
            Hit(
                id="doc-1",
                title="Stub title",
                text="Stub body",
                score=0.5,
                metadata={"topic": "demo"},
            )
        ]


def sample_document(**overrides: object) -> Document:
    fields: dict[str, object] = {
        "id": "doc-1",
        "title": "A title",
        "text": "Some body text.",
        "source": "fixture",
        "url": None,
        "date": None,
        "license": "MIT",
        "metadata": {},
    }
    fields.update(overrides)
    return Document.model_validate(fields)
