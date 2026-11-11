"""Build and describe an index directory.

An index directory holds three files:

- `index.faiss`: a flat inner-product index of L2-normalized vectors (cosine similarity)
- `docs.jsonl`: the documents, in the same order as the vectors
- `manifest.json`: the model, revision, dataset and build time

The API and the CLI both refuse a directory whose manifest names a different model or
revision than the process is configured for.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ssv2.config import Settings
from ssv2.documents import Document, passage_text
from ssv2.embedding import Embedder, l2_normalize

MANIFEST_FILE = "manifest.json"
DOCS_FILE = "docs.jsonl"
FAISS_FILE = "index.faiss"


class IndexManifestError(RuntimeError):
    """The index on disk was not built for this process's embedding model."""


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    document_count: int = Field(ge=0)
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int = Field(gt=0)
    metric: str = "inner_product"
    normalized: bool = True
    built_at: datetime
    query_instruction: str


def build_index(
    documents: Iterable[Document],
    embedder: Embedder,
    output_dir: Path,
    dataset: str,
) -> IndexManifest:
    """Embed `documents`, write the index directory, and return its manifest.

    Duplicate ids are rejected: a later lookup by id would be ambiguous.
    """
    materialized: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        if document.id in seen:
            raise ValueError(f"duplicate document id {document.id!r}")
        seen.add(document.id)
        materialized.append(document)
    if not materialized:
        raise ValueError("no documents to index")

    vectors = np.asarray(
        embedder.embed_documents([passage_text(document) for document in materialized]),
        dtype=np.float32,
    )
    if vectors.shape != (len(materialized), embedder.dimension):
        raise ValueError(
            f"embedder returned shape {vectors.shape}, expected "
            f"({len(materialized)}, {embedder.dimension})"
        )
    normalized = np.ascontiguousarray(l2_normalize(vectors))

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss_index = faiss.IndexFlatIP(embedder.dimension)
    faiss_index.add(normalized)
    faiss.write_index(faiss_index, str(output_dir / FAISS_FILE))
    (output_dir / DOCS_FILE).write_text(
        "".join(document.model_dump_json() + "\n" for document in materialized),
        encoding="utf-8",
    )
    manifest = IndexManifest(
        dataset=dataset,
        document_count=len(materialized),
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        embedding_dimension=embedder.dimension,
        metric="inner_product",
        normalized=True,
        built_at=datetime.now(UTC),
        query_instruction=embedder.query_instruction,
    )
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(index_dir: Path) -> IndexManifest:
    path = index_dir / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No index at {index_dir}. Build one with `ssv2 index scifact` or `ssv2 index jsonl`."
        )
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def assert_manifest_matches(manifest: IndexManifest, settings: Settings) -> None:
    """Raise if this process would embed queries with a different model than built the index."""
    model_ok = manifest.embedding_model == settings.embedding_model
    revision_ok = manifest.embedding_model_revision == settings.embedding_model_revision
    if model_ok and revision_ok:
        return
    raise IndexManifestError(
        f"Index was built with {manifest.embedding_model} revision "
        f"{manifest.embedding_model_revision}, but this process is configured for "
        f"{settings.embedding_model} revision {settings.embedding_model_revision}. "
        "Rebuild the index, or point SSV2_INDEX_DIR at one built with the configured model."
    )
