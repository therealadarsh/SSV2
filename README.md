# SSV2: Semantic Search, version 2

A retrieval service you can trust because it measures itself. It answers search queries over
a document collection with ranked, scored results. Every change is checked against a
labelled benchmark in CI, so "search got better" is a number, not a feeling.

> **Status: planning.** No code yet. Start with [docs/PLAN.md](docs/PLAN.md).

## What it will do

- `POST /v1/search` returns the top matching documents, with scores and metadata
- A generic ingestion pipeline: any corpus goes in through one document schema
- An evaluation harness that reports recall@k, nDCG@k and MRR on every PR
- A minimal web UI for trying queries
- Later: hybrid search, reranking, then RAG (cited answers from an LLM)

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
| [docs/KICKOFF_PROMPT.md](docs/KICKOFF_PROMPT.md) | The prompt to start a new Claude Code session on this repo |
| [CLAUDE.md](CLAUDE.md) | Conventions that any coding agent working here must follow |

## License

[MIT](LICENSE)
