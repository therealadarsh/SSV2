# SSV2: Semantic Search, version 2

A retrieval service you can trust because it measures itself. It answers search queries over
a document collection with ranked, scored results. Every change is checked against a
labelled benchmark in CI, so "search got better" is a number, not a feeling.

> **Status: Phase 1 (index and search).** `ssv2 search` and `POST /v1/search` answer from a
> dense index. The evaluation harness is Phase 2. See [docs/PLAN.md](docs/PLAN.md) and
> [docs/HANDOFF.md](docs/HANDOFF.md).

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Docker. uv installs Python 3.12 if you don't have it.

```bash
make install     # .venv from uv.lock (CPU-only PyTorch) + git pre-commit hook
make check       # lint + typecheck + unit tests: what CI's first three jobs run
make index       # download SciFact and build indexes/scifact (needs the network)
make run         # API with auto-reload: http://localhost:8000/docs
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
| `GET /health` | Liveness: the process is up (used by the Docker `HEALTHCHECK`) |
| `GET /ready` | Readiness: model and index are loaded and the manifest matches; `503` otherwise |
| `GET /v1/info` | Dataset, document count, embedding model, revision, build time |
| `POST /v1/search` | `{"query": str, "k": int, "filters"?: {str: str}}` → hits with id, title, text, score, metadata |

Configuration comes from `SSV2_*` environment variables, all defined in
[src/ssv2/config.py](src/ssv2/config.py).

## CI/CD

- **CI** ([ci.yml](.github/workflows/ci.yml)), on every PR and push to `main`: `lint`,
  `typecheck` and `test` in parallel, then `docker` (build, smoke test, Trivy scan). Each step
  is a `make` target, so any failure reproduces locally.
- **Release** ([release.yml](.github/workflows/release.yml)), on a `v*.*.*` tag: re-runs CI,
  then pushes the image to `ghcr.io` with an SBOM and a signed provenance attestation.
- **Dependabot** ([dependabot.yml](.github/dependabot.yml)) bumps actions, base images and
  `uv.lock` weekly, after a 7-day cooldown.

## What it does

- `POST /v1/search` returns the top matching documents, with scores and metadata
- A generic ingestion pipeline: any corpus goes in through one document schema
- Still to come: an evaluation harness (recall@k, nDCG@k, MRR) on every PR, a minimal
  search page, then hybrid search, reranking, and RAG

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
