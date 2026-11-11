"""Dense retrieval over a saved index.

Filtering is exact: when filters are set, every document is scored and then filtered.
That is cheap at SciFact size and stays correct. Approximate filtering can wait until
the corpus is large enough that scoring all of it matters.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
from pydantic import BaseModel, Field

from ssv2.config import Settings
from ssv2.documents import Document
from ssv2.embedding import Embedder, l2_normalize
from ssv2.index import (
    DOCS_FILE,
    FAISS_FILE,
    IndexManifest,
    IndexManifestError,
    assert_manifest_matches,
    load_manifest,
)

MIN_K = 1
MAX_K = 100


class Hit(BaseModel):
    id: str
    title: str
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class Retriever(Protocol):
    info: IndexManifest

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]: ...


class DenseRetriever:
    """Inner product on the normalized vectors in an index directory."""

    def __init__(self, index_dir: Path, embedder: Embedder, settings: Settings) -> None:
        manifest = load_manifest(index_dir)
        assert_manifest_matches(manifest, settings)
        if manifest.embedding_dimension != embedder.dimension:
            raise IndexManifestError(
                f"Index dimension is {manifest.embedding_dimension}, "
                f"but the embedder reports {embedder.dimension}."
            )
        docs_path = index_dir / DOCS_FILE
        faiss_path = index_dir / FAISS_FILE
        if not docs_path.is_file() or not faiss_path.is_file():
            raise FileNotFoundError(f"Index at {index_dir} is missing {DOCS_FILE} or {FAISS_FILE}.")
        documents = [
            Document.model_validate_json(line)
            for line in docs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vector_index = faiss.read_index(str(faiss_path))
        if vector_index.ntotal != len(documents) or len(documents) != manifest.document_count:
            raise IndexManifestError(
                f"Index at {index_dir} is inconsistent: manifest says {manifest.document_count}, "
                f"docs.jsonl has {len(documents)}, faiss has {vector_index.ntotal}."
            )
        self.info = manifest
        self._documents = documents
        self._index = vector_index
        self._embedder = embedder

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        total = int(self._index.ntotal)
        if total == 0 or k <= 0:
            return []
        active_filters = {key: value for key, value in (filters or {}).items()}
        fetch = total if active_filters else min(k, total)
        embedded = l2_normalize(self._embedder.embed_query(query))
        query_vector = np.ascontiguousarray(embedded, dtype=np.float32).reshape(1, -1)
        scores, ids = self._index.search(query_vector, fetch)
        hits: list[Hit] = []
        for score, row in zip(scores[0], ids[0], strict=True):
            index = int(row)
            if index < 0 or index >= len(self._documents):
                continue
            document = self._documents[index]
            if active_filters and not _matches(document, active_filters):
                continue
            hits.append(
                Hit(
                    id=document.id,
                    title=document.title,
                    text=document.text,
                    score=float(score),
                    metadata=dict(document.metadata),
                )
            )
            if len(hits) == k:
                break
        return hits


def _matches(document: Document, filters: Mapping[str, str]) -> bool:
    return all(document.metadata.get(key) == value for key, value in filters.items())
