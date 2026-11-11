# SSV2: Semantic Search, version 2

A retrieval service you can trust because it measures itself. It answers search queries over
a document collection with ranked, scored results. Every change is checked against a
labelled benchmark in CI, so "search got better" is a number, not a feeling.

> **Status: v0.1.0 (MVP).** Search at `/` and `POST /v1/search`. `make eval` scores SciFact
> and CI fails if nDCG@10 drops below 0.6827. Latest scorecard: nDCG@10 0.7127, recall@10
> 0.8362, MRR@10 0.6822 (0-1 scale, 300 test queries). Warmed SciFact search p95 is 9.6 ms.
> See [docs/PLAN.md](docs/PLAN.md).

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Docker. uv installs Python 3.12 if you don't have it.

```bash
make install     # .venv from uv.lock (CPU-only PyTorch) + git pre-commit hook
make check       # lint + typecheck + unit tests: what CI's first three jobs run
make index       # download SciFact and build indexes/scifact (needs the network)
make eval        # nDCG@10, recall@10, MRR@10; fails under the floor
make bench       # warmed search latency; the MVP budget is p95 under 200 ms
make run         # API with auto-reload: http://localhost:8000/
ssv2 search "a scientific claim"    # query the index you just built
make smoke       # build the image, run it hardened, query health, ready and search
make scan        # Trivy: fail on fixable HIGH/CRITICAL CVEs in the image
make help        # every target
```

`make index` builds the SciFact benchmark (non-commercial, evaluation only) under
`indexes/scifact`, which is gitignored. `make run` serves whatever `SSV2_INDEX_DIR` points
at; it defaults to that directory, so index first or startup will refuse to become ready.
The Docker image does not contain SciFact. It ships a four-document built-in corpus so the
container can answer a search with no network.

| Endpoint | Meaning |
| --- | --- |
| `GET /` | Search page: a query box that calls `POST /v1/search` and lists the hits |
| `GET /health` | Liveness: the process is up (used by the Docker `HEALTHCHECK`) |
| `GET /ready` | Readiness: model and index are loaded and the manifest matches; `503` otherwise |
| `GET /v1/info` | Dataset, document count, embedding model, revision, build time |
| `POST /v1/search` | `{"query": str, "k": int, "filters"?: {str: str}}` → hits with id, title, text, score, metadata |

Configuration comes from `SSV2_*` environment variables, all defined in
[src/ssv2/config.py](src/ssv2/config.py).

## CI/CD

- **CI** ([ci.yml](.github/workflows/ci.yml)), on every PR and push to `main`: `lint`,
  `typecheck` and `test` in parallel, then `eval` and `docker` (build, smoke test, Trivy scan).
  Each step is a `make` target, so any failure reproduces locally. `eval` caches the Hugging
  Face model and `indexes/scifact`.
- **Release** ([release.yml](.github/workflows/release.yml)), on a `v*.*.*` tag: re-runs CI,
  then pushes the image to `ghcr.io` with an SBOM and a signed provenance attestation.
  The package version is `0.1.0`. After the MVP commit is on `main`, publish with
  `git tag v0.1.0 && git push origin v0.1.0`. Tag protection rejects that push until the
  commit is on `main`, so the tag is not created from a feature branch.
- **Dependabot** ([dependabot.yml](.github/dependabot.yml)) bumps actions, base images and
  `uv.lock` weekly, after a 7-day cooldown.

## What it does

- `POST /v1/search` returns the top matching documents, with scores and metadata
- A generic ingestion pipeline: any corpus goes in through one document schema
- `make eval` reports nDCG@10, recall@10 and MRR@10 on SciFact and fails under the floor
- A search page at `/`, and one JSON log line per request with its duration
- Optional hybrid search and cross-encoder reranking (`SSV2_RETRIEVER=hybrid` or `rerank`). On SciFact both score below the dense default, so the default stays dense
- `ssv2 index arxiv` indexes CC BY 4.0 computer-science papers from a pinned metadata shard
- Still to come: RAG

## Where it comes from

SSV2 is a rewrite of [Semantic-Search](https://github.com/therealadarsh/Semantic-Search).
That repo's modernization branch contains the CI/CD, Docker and testing patterns this
project starts from, explained in its
[CI/CD guide](https://github.com/therealadarsh/Semantic-Search/blob/claude/jolly-planck-wlqzqt/docs/CICD_GUIDE.md).

## Docs

| File | What's in it |
| --- | --- |
| [docs/PLAN.md](docs/PLAN.md) | Goals, MVP definition, architecture, data, phases with gates |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decisions made so far and why |
| [docs/REPO_SETTINGS.md](docs/REPO_SETTINGS.md) | One-time GitHub settings that protect this repo |
| [docs/KICKOFF_PROMPT.md](docs/KICKOFF_PROMPT.md) | The prompt that started Phase 0 |
| [docs/HANDOFF.md](docs/HANDOFF.md) | The prompt for Phase 2, once this phase is on `main` |
| [CLAUDE.md](CLAUDE.md) | Conventions that any coding agent working here must follow |

## License

[MIT](LICENSE)
