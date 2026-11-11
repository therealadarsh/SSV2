# Decisions log

Newest first. Add a row whenever a choice would be hard to reverse or someone might later
ask "why did we do it this way?". Eval-driven decisions must cite the scorecard numbers.

| Date | Decision | Why | Alternatives considered |
| --- | --- | --- | --- |
| 2026-11-11 | Python 3.12, FastAPI, `uv` with a committed lockfile, `src/` layout | Familiar from Semantic-Search; a lockfile pins transitive dependencies; `src/` stops tests from accidentally importing the working tree | Plain requirements.txt (no transitive pins); Poetry (slower, less common now) |
| 2026-11-11 | In-memory FAISS flat index for the MVP | Corpora are under 100 MB of vectors; exact search is fast and simple | pgvector or Qdrant (needed later for live updates or more than ~1M docs) |
| 2026-11-11 | Embedding model `BAAI/bge-small-en-v1.5`, pinned to a revision | MIT license, 384 dimensions, runs well on CPU, strong on BEIR | all-MiniLM-L6-v2 (weaker); larger models (slower on CPU). Revisit using eval numbers. |
| 2026-11-11 | SciFact (BEIR) for evaluation; arXiv CS slice (CC BY 4.0 papers only) as the realistic corpus | SciFact has expert relevance labels and published baselines; arXiv metadata is CC0, current, and has dates and categories | AG News (non-commercial, 2004); MS MARCO (non-commercial, large); Wikipedia (share-alike) |
| 2026-11-11 | Start a new repo instead of evolving Semantic-Search | A clean design with no legacy scripts; Semantic-Search stays as the reference | Evolve in place (keeps history but drags old structure along) |
