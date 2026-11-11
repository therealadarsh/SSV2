# Working on SSV2

Read [docs/PLAN.md](docs/PLAN.md) before starting any task. Work happens phase by phase,
and a phase isn't done until its gate passes.

## Rules

- **Never push to `main`.** Work on a branch and open a PR. `main` is protected.
- **Measure retrieval changes.** Any change to ingestion, embedding, indexing or ranking must
  include the before and after `make eval` numbers in the PR description. Record notable
  results in docs/DECISIONS.md.
- **No silent model mismatch.** The index manifest records the embedding model and revision,
  and the API must refuse an index that doesn't match its configuration.
- **Pin everything:** dependencies (`uv.lock`), model and dataset revisions, and Docker base
  images. In GitHub Actions, prefer full commit SHAs with a version comment.
- **Respect data licenses.** Record each dataset's license in its loader. SciFact is for
  evaluation only (non-commercial). From arXiv, ingest only papers licensed CC BY 4.0.
- **No secrets in the repo.** API keys come from environment variables. `.env` is gitignored.
- **Keep CI and local commands the same:** every CI step must be runnable through a
  `make` target.

## Commands (once Phase 0 exists)

```bash
make install     # uv sync, pre-commit install
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict
make test        # pytest, unit tests only (fast, no model download)
make check       # lint + typecheck + test
make index       # build the index for $DATASET (default: scifact)   [Phase 1]
make eval        # scorecard: nDCG@10, recall@10, MRR@10              [Phase 2]
make run         # API with reload at http://localhost:8000
make smoke       # build the image, run it, query it
make scan        # Trivy scan of the image (fails on fixable HIGH/CRITICAL)
```

Behind a TLS-intercepting proxy (as in Claude Code cloud sessions), pass the proxy CA to the
build: `make smoke DOCKER_BUILD_FLAGS="--network host --build-arg HTTPS_PROXY=$HTTPS_PROXY
--build-arg https_proxy=$HTTPS_PROXY --secret id=extra_ca,src=/root/.ccr/ca-bundle.crt"`.

## Layout (target)

```
src/ssv2/
  config.py        # settings from env vars, one place
  documents.py     # Document schema
  loaders/         # one module per dataset -> Iterator[Document]
  embedding.py     # Embedder interface + sentence-transformers implementation
  index.py         # build / save / load the index directory (faiss + docs + manifest)
  retrieval.py     # Retriever interface + dense implementation
  evals/           # metrics.py, run.py
  api/             # FastAPI app, routes, schemas, static UI
  cli.py           # ssv2 index | search | eval
tests/             # unit (fast, fakes) and integration (marked, real model)
```

## Network note for Claude Code cloud sessions

Model, dataset and CPU-only PyTorch downloads need `huggingface.co`, its file hosts,
`download.pytorch.org` and `download-r2.pytorch.org` (where the CPU wheels actually live).
Docker builds also need `pkg-containers.githubusercontent.com` (ghcr.io image layers) and
`deb.debian.org` (security updates). If Docker Hub answers `429 Too Many Requests`, start
`dockerd --registry-mirror https://mirror.gcr.io`. All of these hosts are allowed in the
owner's "Dev" cloud environment, so run sessions there.
