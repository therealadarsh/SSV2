"""Pick the retriever named by settings. The default is the dense index."""

from pathlib import Path

from ssv2.config import Settings
from ssv2.embedding import Embedder
from ssv2.retrieval import DenseRetriever, Retriever


def build_retriever(index_dir: Path, embedder: Embedder, settings: Settings) -> Retriever:
    dense = DenseRetriever(index_dir, embedder, settings)
    if settings.retriever == "dense":
        return dense
    if settings.retriever == "hybrid":
        from ssv2.hybrid import HybridRetriever

        return HybridRetriever(dense)
    from ssv2.rerank import CrossEncoderReranker, RerankedRetriever

    return RerankedRetriever(dense, CrossEncoderReranker(settings))
