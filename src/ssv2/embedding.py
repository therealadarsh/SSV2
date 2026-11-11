"""Embedders.

Queries for `bge-small-en-v1.5` take the retrieval instruction from the model card.
Passages do not. Vectors are not normalized here; the index normalizes both sides so
every embedder, including fakes, is compared by cosine.
"""

import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from ssv2.config import Settings

# Trailing space is part of the published instruction.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Anything that turns text into vectors. Tests pass a fake; the CLI uses the real model."""

    model_id: str
    revision: str
    dimension: int
    query_instruction: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def format_query(text: str) -> str:
    """Prepend the BGE retrieval instruction. Do not use this for passages."""
    return QUERY_INSTRUCTION + text


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row (or a single vector) to unit length. A zero vector stays zero."""
    matrix = np.asarray(vectors, dtype=np.float32)
    single = matrix.ndim == 1
    if single:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"expected a vector or a matrix, got shape {vectors.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, np.float32(1e-12))
    normalized = (matrix / norms).astype(np.float32, copy=False)
    if single:
        return cast(np.ndarray, normalized[0])
    return cast(np.ndarray, normalized)


def prepare_hf_home() -> Path:
    """Return a writable Hugging Face cache directory.

    The container runs with a read-only root filesystem. Model files live under `HF_HOME`,
    which is not writable, and the hub library creates lock files next to them. When the
    configured cache cannot be written, copy it into a temp directory and point the
    process at the copy. A writable cache is left where it is.
    """
    raw = os.environ.get("HF_HOME")
    if raw is None:
        return Path.home() / ".cache" / "huggingface"
    cache = Path(raw)
    if not cache.exists() or os.access(cache, os.W_OK):
        return cache
    destination = Path(tempfile.mkdtemp(prefix="ssv2-hf-"))
    shutil.copytree(cache, destination, dirs_exist_ok=True, copy_function=shutil.copyfile)
    # copytree copies the source mode, which is the non-writable one we are escaping.
    for directory, _dirnames, filenames in os.walk(destination):
        os.chmod(directory, 0o700)
        for filename in filenames:
            os.chmod(Path(directory) / filename, 0o600)
    os.environ["HF_HOME"] = str(destination)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(destination / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(destination / "transformers")
    return destination


class SentenceTransformersEmbedder:
    """The configured sentence-transformers model, pinned to a revision when loaded from the hub."""

    def __init__(self, settings: Settings) -> None:
        prepare_hf_home()
        # Imported here so `ssv2 --help` and the unit tests do not import PyTorch.
        from sentence_transformers import SentenceTransformer

        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        local_dir = settings.embedding_model_dir
        model_name = str(local_dir) if local_dir is not None else settings.embedding_model
        self._model = SentenceTransformer(
            model_name,
            revision=None if local_dir is not None else settings.embedding_model_revision,
            device="cpu",
            local_files_only=offline or local_dir is not None,
            trust_remote_code=False,
        )
        self.model_id = settings.embedding_model
        self.revision = settings.embedding_model_revision
        self.query_instruction = QUERY_INSTRUCTION
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"{settings.embedding_model} did not report an embedding dimension")
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        # prompt="" disables a default prompt saved in the model config. Passages get none.
        encoded = self._model.encode(
            list(texts),
            prompt="",
            batch_size=32,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(encoded, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        encoded = self._model.encode(
            [format_query(text)],
            prompt="",
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        return cast(np.ndarray, matrix[0])
