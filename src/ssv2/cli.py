"""Command line: `ssv2 index` builds an index, `ssv2 search` queries one.

The real model is imported only when a command actually embeds something, so `--help` stays
fast and unit tests can pass a fake embedder.
"""

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from ssv2.config import Settings, get_settings
from ssv2.embedding import Embedder
from ssv2.evals.run import EVAL_K, score_retrieval, write_scorecard
from ssv2.index import IndexManifestError, build_index
from ssv2.latency import percentile
from ssv2.loaders.arxiv import download_arxiv_shard, load_arxiv_shard
from ssv2.loaders.jsonl import load_jsonl
from ssv2.loaders.scifact import (
    download_scifact_corpus,
    download_scifact_qrels,
    download_scifact_queries,
    load_qrels_tsv,
    load_scifact_corpus,
    load_scifact_queries,
)
from ssv2.retrieval import MAX_K, MIN_K, Hit
from ssv2.retrievers import build_retriever

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
        if args.command == "eval":
            return _eval(args, embedder_factory or _real_embedder)
        if args.command == "bench":
            return _bench(args, embedder_factory or _real_embedder)
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

    arxiv = sources.add_parser("arxiv", help="Index CC BY 4.0 computer-science papers from arXiv")
    arxiv.add_argument(
        "--path", type=Path, default=None, help="Metadata parquet (default: pinned shard)"
    )
    arxiv.add_argument("--limit", type=int, default=2000, help="Maximum papers to index")
    arxiv.add_argument(
        "--category-prefix", default="cs.", help="Keep categories that start with this"
    )
    arxiv.add_argument("--output", type=Path, default=None, help="Index directory")

    search = subcommands.add_parser("search", help="Query an index")
    search.add_argument("query", help="The query text")
    search.add_argument("-k", type=int, default=10, help=f"Number of hits ({MIN_K}-{MAX_K})")
    search.add_argument(
        "--index", type=Path, default=None, help="Index directory (default: SSV2_INDEX_DIR)"
    )
    search.add_argument("--json", action="store_true", help="Print hits as JSON")

    evaluate = subcommands.add_parser("eval", help="Score the index on the SciFact test split")
    evaluate.add_argument("--index", type=Path, default=None, help="Index directory")
    evaluate.add_argument(
        "--queries", type=Path, default=None, help="Queries parquet (default: pinned SciFact)"
    )
    evaluate.add_argument(
        "--qrels", type=Path, default=None, help="Qrels TSV (default: pinned test split)"
    )
    evaluate.add_argument("--output", type=Path, default=None, help="Scorecard JSON path")
    evaluate.add_argument("-k", type=int, default=EVAL_K, help=f"Cutoff (default {EVAL_K})")
    evaluate.add_argument(
        "--retriever",
        choices=("dense", "hybrid", "rerank"),
        default=None,
        help="Override SSV2_RETRIEVER for this run",
    )

    bench = subcommands.add_parser("bench", help="Measure search latency after one warmup query")
    bench.add_argument("--index", type=Path, default=None, help="Index directory")
    bench.add_argument("--query", default="a scientific claim", help="Query to repeat")
    bench.add_argument("-n", type=int, default=50, help="Timed searches, after one warmup")
    bench.add_argument("-k", type=int, default=10, help=f"Hits per search ({MIN_K}-{MAX_K})")
    bench.add_argument(
        "--max-p95-ms", type=float, default=200.0, help="Exit 1 when p95 is above this"
    )
    return parser


def _index(args: argparse.Namespace, embedder_factory: EmbedderFactory) -> int:
    settings = get_settings()
    if args.source == "scifact":
        dataset = "scifact"
        documents = list(load_scifact_corpus(download_scifact_corpus()))
    elif args.source == "arxiv":
        dataset = "arxiv"
        if args.limit < 1:
            print("limit must be at least 1", file=sys.stderr)
            return 2
        shard = args.path if args.path is not None else download_arxiv_shard()
        documents = list(
            load_arxiv_shard(shard, category_prefix=args.category_prefix, limit=args.limit)
        )
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
    hits = build_retriever(settings.index_dir, embedder, settings).search(query, args.k)
    if args.json:
        print(json.dumps({"hits": [hit.model_dump(mode="json") for hit in hits]}, indent=2))
    else:
        _print_hits(hits)
    return 0


def _eval(args: argparse.Namespace, embedder_factory: EmbedderFactory) -> int:
    if args.k < MIN_K or args.k > MAX_K:
        print(f"k must be between {MIN_K} and {MAX_K}", file=sys.stderr)
        return 2
    settings = get_settings()
    if args.index is not None:
        settings = settings.model_copy(update={"index_dir": args.index})
    queries = load_scifact_queries(
        args.queries if args.queries is not None else download_scifact_queries()
    )
    qrels = load_qrels_tsv(args.qrels if args.qrels is not None else download_scifact_qrels())
    if args.retriever is not None:
        settings = settings.model_copy(update={"retriever": args.retriever})
    retriever = build_retriever(settings.index_dir, embedder_factory(settings), settings)
    scorecard = score_retrieval(retriever, queries, qrels, dataset="scifact", k=args.k)
    destination = args.output if args.output is not None else settings.index_dir / "scorecard.json"
    write_scorecard(scorecard, destination)
    print(scorecard.model_dump_json(indent=2))
    if not scorecard.passed:
        print(
            f"nDCG@{scorecard.k} {scorecard.ndcg_at_k:.4f} is below the floor "
            f"{scorecard.ndcg_at_10_floor:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


def _bench(args: argparse.Namespace, embedder_factory: EmbedderFactory) -> int:
    if args.n < 1:
        print("n must be at least 1", file=sys.stderr)
        return 2
    if args.k < MIN_K or args.k > MAX_K:
        print(f"k must be between {MIN_K} and {MAX_K}", file=sys.stderr)
        return 2
    query = str(args.query).strip()
    if not query:
        print("query must not be blank", file=sys.stderr)
        return 2
    if args.max_p95_ms <= 0:
        print("max p95 must be positive", file=sys.stderr)
        return 2
    settings = get_settings()
    if args.index is not None:
        settings = settings.model_copy(update={"index_dir": args.index})
    retriever = build_retriever(settings.index_dir, embedder_factory(settings), settings)
    retriever.search(query, args.k)
    samples: list[float] = []
    for _ in range(args.n):
        started = time.perf_counter()
        retriever.search(query, args.k)
        samples.append((time.perf_counter() - started) * 1000)
    report = {
        "n": args.n,
        "k": args.k,
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "max_ms": max(samples),
    }
    print(json.dumps(report, indent=2))
    if report["p95_ms"] > args.max_p95_ms:
        print(
            f"p95 {report['p95_ms']:.1f} ms is above {args.max_p95_ms:.1f} ms",
            file=sys.stderr,
        )
        return 1
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
