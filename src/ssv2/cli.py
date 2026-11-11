"""Command line: `ssv2 index` builds an index, `ssv2 search` queries one.

The real model is imported only when a command actually embeds something, so `--help` stays
fast and unit tests can pass a fake embedder.
"""

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ssv2.config import Settings, get_settings
from ssv2.embedding import Embedder
from ssv2.index import IndexManifestError, build_index
from ssv2.loaders.jsonl import load_jsonl
from ssv2.loaders.scifact import download_scifact_corpus, load_scifact_corpus
from ssv2.retrieval import MAX_K, MIN_K, DenseRetriever, Hit

_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
EmbedderFactory = Callable[[Settings], Embedder]


def main(
    argv: Sequence[str] | None = None, *, embedder_factory: EmbedderFactory | None = None
) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # Each invocation reads the environment itself. The settings cache is for the API process.
    get_settings.cache_clear()
    try:
        if args.command == "search":
            return _search(args, embedder_factory or _real_embedder)
        return _index(args, embedder_factory or _real_embedder)
    except (FileNotFoundError, IndexManifestError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ssv2", description="Build a dense index and search it.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    index = subcommands.add_parser("index", help="Embed a corpus and write an index directory")
    sources = index.add_subparsers(dest="source", required=True)

    scifact = sources.add_parser("scifact", help="Download the pinned SciFact corpus and index it")
    scifact.add_argument(
        "--output", type=Path, default=None, help="Index directory (default: indexes/scifact)"
    )

    jsonl = sources.add_parser("jsonl", help="Index a JSONL file of documents")
    jsonl.add_argument("path", type=Path, help="JSONL file in the document schema")
    jsonl.add_argument("--dataset", required=True, help="Name recorded in the manifest")
    jsonl.add_argument(
        "--output", type=Path, default=None, help="Index directory (default: indexes/<dataset>)"
    )

    search = subcommands.add_parser("search", help="Query an index")
    search.add_argument("query", help="The query text")
    search.add_argument("-k", type=int, default=10, help=f"Number of hits ({MIN_K}-{MAX_K})")
    search.add_argument(
        "--index", type=Path, default=None, help="Index directory (default: SSV2_INDEX_DIR)"
    )
    search.add_argument("--json", action="store_true", help="Print hits as JSON")
    return parser


def _index(args: argparse.Namespace, embedder_factory: EmbedderFactory) -> int:
    settings = get_settings()
    if args.source == "scifact":
        dataset = "scifact"
        documents = list(load_scifact_corpus(download_scifact_corpus()))
    else:
        dataset = str(args.dataset)
        if _DATASET_NAME.fullmatch(dataset) is None:
            raise ValueError(f"dataset name must be a plain slug, not {dataset!r}")
        documents = list(load_jsonl(args.path))
    output = args.output
    if output is None:
        output = settings.index_dir if dataset == "scifact" else Path("indexes") / dataset
    embedder = embedder_factory(settings)
    manifest = build_index(documents, embedder, output, dataset=dataset)
    print(
        f"Indexed {manifest.document_count} documents from {manifest.dataset} into {output}\n"
        f"model {manifest.embedding_model} revision {manifest.embedding_model_revision}"
    )
    return 0


def _search(args: argparse.Namespace, embedder_factory: EmbedderFactory) -> int:
    if args.k < MIN_K or args.k > MAX_K:
        print(f"k must be between {MIN_K} and {MAX_K}", file=sys.stderr)
        return 2
    query = str(args.query).strip()
    if not query:
        print("query must not be blank", file=sys.stderr)
        return 2
    settings = get_settings()
    if args.index is not None:
        settings = settings.model_copy(update={"index_dir": args.index})
    embedder = embedder_factory(settings)
    hits = DenseRetriever(settings.index_dir, embedder, settings).search(query, args.k)
    if args.json:
        print(json.dumps({"hits": [hit.model_dump(mode="json") for hit in hits]}, indent=2))
    else:
        _print_hits(hits)
    return 0


def _print_hits(hits: list[Hit]) -> None:
    if not hits:
        print("No hits.")
        return
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}\t{hit.score:.4f}\t{hit.id}\t{hit.title}")
        snippet = " ".join(hit.text.split())
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        print(f"\t{snippet}")


def _real_embedder(settings: Settings) -> Embedder:
    from ssv2.embedding import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(settings)
