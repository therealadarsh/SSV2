# Handoff prompt

Phase 1 is on `main` when you use this. Start a new Cursor cloud agent on
`https://github.com/therealadarsh/SSV2` in the **Dev** environment (it allows Hugging Face and
PyTorch downloads), on a branch, never on `main`, then paste:

---

Phase 1 is merged. SciFact can be indexed with `BAAI/bge-small-en-v1.5` at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. `ssv2 search` and `POST /v1/search` share one
dense retriever. Do **Phase 2 only** from `docs/PLAN.md`: the evaluation harness. Stop when
its gate passes. Do not start the search page, logging, or the release.

1. Read `CLAUDE.md`, `docs/PLAN.md`, `docs/DECISIONS.md`, and `docs/HANDOFF.md`. Then read
   the retrieval path so the harness calls it instead of reimplementing it:
   `src/ssv2/documents.py` (`passage_text`), `src/ssv2/embedding.py` (the query instruction),
   `src/ssv2/loaders/scifact.py` (the pinned corpus), `src/ssv2/index.py`,
   `src/ssv2/retrieval.py`, and `src/ssv2/cli.py`.
2. Add metric functions for recall@k, nDCG@k and MRR@k, with unit tests on hand-worked
   examples. No model and no network in those tests. Use the standard graded definition so
   the numbers match MTEB:
   - gain is `2^relevance - 1`
   - the discount at 1-based rank `r` is `log2(r + 1)`
   - nDCG is DCG divided by the ideal DCG, and 0 when nothing is relevant
   - recall@k is the fraction of relevant documents that appear in the top k
   - MRR is `1 / rank` of the first relevant hit, and 0 when there is none
   - the scorecard averages those, unweighted, over the test queries
3. Load queries from the same pinned SciFact repo
   (`BeIR/scifact` @ `b3b5335604bf5ee3c4447671af975ea25143d4f5`,
   `queries/queries-00000-of-00001.parquet`). Load labels from
   `BeIR/scifact-qrels` @ `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`, file `test.tsv`
   (header `query-id`, `corpus-id`, `score`). Score only queries that appear in `test.tsv`.
   The queries file also contains train queries; do not score those. SciFact is
   non-commercial. Do not put the corpus or the qrels in the Docker image.
4. `ssv2 eval` and `make eval` run the existing retriever over that test split and write a
   JSON scorecard with nDCG@10, recall@10 and MRR@10. `make test` must stay fast: no model
   download. Building the SciFact index is `make index` (already implemented).
5. Add an `eval` CI job that runs `make eval`. Cache the Hugging Face model and the built
   index between runs. The job fails when nDCG@10 is below the floor. The floor is a
   constant in the eval code, so the same `make eval` fails locally. When the job exists,
   add `eval` to the required checks described in `docs/REPO_SETTINGS.md`.
6. The gate is the published MTEB score for this exact model, from its model card:
   **nDCG@10 = 71.275**, which is **0.7128** on the 0–1 scale (use one scale everywhere
   and say which). Recall@10 83.622 and MRR@10 68.219 are sanity checks, not gates.
   A one-off run of this retriever on the 300 SciFact test queries already scored
   nDCG@10 0.7127, recall@10 0.8362, MRR@10 0.6822. `make eval` should reproduce that.
   Do not change the query instruction or the passage format unless your harness
   disagrees with those numbers. If it does, change one thing, remeasure, and keep the
   better setup. If nDCG@10 is within 0.02 of 0.7128, set the CI floor to the measured
   value minus 0.03, so a real drop fails and noise does not. Record the scorecard,
   the floor, and the choice in `docs/DECISIONS.md`.
7. This is the first eval, so there is no "before" number. The PR description should
   include the scorecard anyway. Any later change to ingestion, embedding, indexing or
   ranking needs before and after `make eval` numbers in the PR, per `CLAUDE.md`.
8. Run `make check` and `make eval` locally and show the results. Push the branch and open
   a PR. Stop after Phase 2's gate.

---

## What Phase 1 left in place

- `ssv2 index scifact` downloads the pinned corpus and writes `indexes/scifact/`
  (`index.faiss`, `docs.jsonl`, `manifest.json`). `ssv2 index jsonl PATH --dataset NAME`
  indexes any corpus already in the document schema.
- `ssv2 search "..." -k 10 --json` queries `SSV2_INDEX_DIR` (default `indexes/scifact`).
- Startup loads the index and refuses a manifest whose model or revision disagrees with
  `SSV2_EMBEDDING_MODEL` and `SSV2_EMBEDDING_MODEL_REVISION`.
- The image contains a four-document built-in index so `/ready` and `/v1/search` work
  offline. It does not contain SciFact. Smoke-test the built-in corpus, and run `make eval`
  against a locally built SciFact index.
- `k` on the API and the CLI is 1–100. Eval at 10 fits.
