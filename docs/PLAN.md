# SSV2 plan

## Goal

Build a small, production-quality retrieval service that measures its own quality, and learn
modern ML engineering and system design while doing it. Build order: retrieval first, then
evaluation, then better retrieval, then generation (RAG).

### Non-goals for the MVP

- No LLM-generated answers (that's Phase 5)
- No user accounts, multi-tenancy or document upload UI
- No GPU, Kubernetes or vector database. One CPU container with an in-memory index is
  enough at this size (see "Back-of-envelope" below).

## MVP: done when all of these are true

1. `POST /v1/search` takes `{"query": str, "k": int, "filters"?: {...}}` and returns hits
   with `id`, `title`, `text`, `score` and `metadata`. p95 latency is under 200 ms on a laptop CPU.
2. `GET /health` (liveness) and `GET /ready` (model and index loaded) both work.
3. `GET /v1/info` reports the index manifest: dataset, document count, embedding model and
   revision, and build time.
4. A minimal web page at `/` searches and shows results.
5. `make eval` reports nDCG@10, recall@10 and MRR@10 on SciFact. CI fails if nDCG@10
   drops below the agreed floor.
6. CI runs lint, type checks, unit tests, eval, Docker build and a container smoke test. Tags
   publish an image to GHCR.
7. The README explains how to run, test, evaluate and release.

## Architecture

```
OFFLINE (make index / Docker build stage)            ONLINE (every request)
┌──────────┐   ┌──────────┐   ┌─────────┐   ┌───────┐        ┌────────┐   ┌───────────┐
│ Loader   │──▶│ Document │──▶│ Embedder│──▶│ Index │──────▶│ /v1/   │──▶│ Retriever │
│ (per     │   │ schema   │   │ (pinned │   │ dir:  │  read  │ search │   │ (dense;   │
│ dataset) │   │          │   │ model)  │   │ faiss │        └────────┘   │ hybrid    │
└──────────┘   └──────────┘   └─────────┘   │ +docs │                     │ later)    │
                                            │ +mani-│        ┌────────┐   └───────────┘
                                            │ fest  │──────▶│ eval   │  same Retriever,
                                            └───────┘        │ harness│  labelled queries
                                                             └────────┘
```

Key rules:

- **One document schema** for every source: `id`, `title`, `text`, `source`, `url`, `date`,
  `license`, `metadata`. Loaders convert each dataset into it, and nothing downstream knows
  which dataset it came from.
- **The index directory carries a manifest** naming the embedding model and revision. The API
  refuses to load an index built with a different model than it is configured for. This
  turns Semantic-Search's silent query/index model mismatch into a startup error.
- **Small interfaces** (`Retriever`, `Embedder`, later `Reranker` and `Generator`) let the API,
  the CLI and the eval harness share one code path, and let tests use fakes.

### Back-of-envelope

- bge-small vectors: 384 dimensions × 4 bytes = 1.5 KB each.
- SciFact: 5,183 docs ≈ 8 MB. arXiv slice: 50k docs ≈ 75 MB. Both fit in RAM, and exact
  (flat) search takes milliseconds.
- Move to an approximate index or a vector database only past roughly 1M documents, or
  when documents must be added while the service runs.

## Data

| Dataset | Role | License | Source |
| --- | --- | --- | --- |
| SciFact (BEIR) | Eval benchmark: 5,183 abstracts, 300 test queries with relevance labels | Upstream SciFact is CC BY-NC 2.0 (non-commercial); the BEIR copy on Hugging Face is tagged CC BY-SA 4.0. Treat it as **non-commercial**. | `BeIR/scifact` and `BeIR/scifact-qrels` on Hugging Face |
| arXiv metadata, one CS category | Realistic corpus to search, with dates and categories for filters | Metadata dataset is CC0; each paper has its own license field. **Keep only CC BY 4.0 papers.** | arXiv metadata snapshot (Kaggle / Hugging Face mirrors) |

Not used: AG News (non-commercial and from 2004), MS MARCO (non-commercial, too big),
BEIR datasets with no published license (FiQA, NFCorpus).

Pin every download to a revision (Hugging Face commit SHA) so builds are reproducible.

## Model

Start with `BAAI/bge-small-en-v1.5`: MIT license, 384 dimensions, CPU-friendly. Hugging
Face revision at planning time: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Keep the model
configurable. Compare alternatives only through the eval, never by reputation.

## What we keep from Semantic-Search, and what we improve

| Keep | Improve |
| --- | --- |
| Multi-stage Dockerfile, CPU-only PyTorch, non-root user, HEALTHCHECK, `$PORT` | Model downloaded at build time from a pinned revision, not committed to git |
| CI: lint → test → Docker build → smoke test; release on `v*` tags to GHCR with provenance | Eval as a CI gate; type checking (mypy or pyright); cached index between CI runs |
| Least-privilege workflow permissions, concurrency, timeouts, Dependabot | Actions pinned to commit SHAs; image vulnerability scan (Trivy) |
| One settings module, model loaded once, structured JSON responses | `src/` package layout, `uv` with a committed lockfile, index manifest, `/ready`, `/v1/info` |
| Fast unit tests with a fake engine; integration tests; smoke test script | Metric functions unit-tested against hand-worked examples |
| Makefile mirroring CI; pre-commit with ruff | Same, plus `make eval` |

## Phases

Each phase ends with a gate. Don't start the next phase until the gate passes.

### Phase 0: Skeleton and pipeline (week 1)
- [x] `src/ssv2/` package, `pyproject.toml`, `uv.lock` with CPU-only PyTorch from the PyTorch index
- [x] FastAPI app with `/health` and `/ready`; ruff, mypy, pytest, pre-commit, Makefile
- [x] Multi-stage Dockerfile; CI (lint, types, tests, Docker build, smoke test, Trivy scan); Dependabot
- [x] Release workflow (tag → GHCR + provenance), copied and adapted from Semantic-Search
- **Gate:** a PR with green CI; after merge, add `lint`, `typecheck`, `test` and `docker` as required checks (docs/REPO_SETTINGS.md)

### Phase 1: Ingestion, index, search (week 2)
- [x] Document schema; SciFact loader; JSONL loader for any corpus
- [x] Embedder (pinned model) and index build: FAISS flat inner-product index on normalized vectors, `docs.jsonl` and `manifest.json`
- [x] `POST /v1/search`, `GET /v1/info`, and the model/manifest check at startup
- [x] CLI: `ssv2 index`, `ssv2 search "..."`
- **Gate:** `ssv2 search` returns sensible results on SciFact, and the API serves them

### Phase 2: Evaluation harness (week 3)
- [x] Metric functions (recall@k, nDCG@k, MRR@k) with unit tests on hand-worked examples
- [x] `ssv2 eval` / `make eval` writes a JSON scorecard
- [x] CI eval job with the index cached between runs; fails below the nDCG@10 floor
- **Gate:** nDCG@10 on SciFact is close to the published score for the chosen model (record it in DECISIONS.md)

### Phase 3: MVP polish (week 4)
- [x] Minimal search page at `/`
- [x] Structured JSON logs with per-request timing; p95 latency measured
- [x] README, first release `v0.1.0`
- **Gate:** every MVP criterion above is true. Package version is `0.1.0`. The `v0.1.0` tag is pushed only after this commit is on `main` (tag protection).

### Phase 4: Better retrieval (weeks 5–6)
- [ ] arXiv CS slice loader (CC BY 4.0 only) with date and category filters
- [ ] Hybrid search: BM25 plus dense results, merged by reciprocal rank fusion
- [ ] Cross-encoder reranking of the top 30
- [ ] Compare each change on the eval: keep only improvements, and record each in DECISIONS.md

### Phase 5: RAG (weeks 7–9)
- [ ] `POST /v1/ask`: retrieve, then send the passages to Claude with citations turned on, and stream the answer
- [ ] Say "I don't know" when retrieval finds nothing relevant
- [ ] Answer-quality eval: faithfulness and correctness, scored by an LLM judge against a rubric
- [ ] Token, cost and latency tracking per request

### Phase 6: Agentic retrieval (later)
- [ ] Expose search as a tool and an MCP server
- [ ] Compare agentic multi-step search with the fixed pipeline on the eval; adopt it only where it wins

## System design practice

Alongside the build, write one page per exercise using six steps: requirements, estimates,
components, interfaces, trade-offs, failure modes.

- 10× the corpus (1M documents)
- 100× the traffic (50 queries per second)
- Daily document updates without downtime
- Multi-tenant, with private corpora per customer
- The LLM provider is down for an hour
