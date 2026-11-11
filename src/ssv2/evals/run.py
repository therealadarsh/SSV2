"""Run the retriever on labelled queries and write a scorecard.

Metrics are on the 0-1 scale. The published MTEB card reports the same nDCG@10 as 71.275,
which is 0.71275 here. A one-off run of this retriever scored 0.7127, so the CI floor sits
0.03 under that measurement.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from ssv2.evals.metrics import mean, mrr_at_k, ndcg_at_k, recall_at_k
from ssv2.retrieval import Hit


class Ranker(Protocol):
    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]: ...


# 0-1 scale. Measured nDCG@10 was 0.7127; three points under that still fails a real drop.
NDCG_AT_10_FLOOR = 0.6827
EVAL_K = 10


class Scorecard(BaseModel):
    scale: str = "0-1"
    dataset: str
    split: str = "test"
    k: int = Field(ge=1)
    query_count: int = Field(ge=1)
    ndcg_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_10_floor: float = NDCG_AT_10_FLOOR
    passed: bool


def score_retrieval(
    retriever: Ranker,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    dataset: str,
    k: int = EVAL_K,
) -> Scorecard:
    """Score every labelled query. Queries that are not in `qrels` are ignored."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not qrels:
        raise ValueError("no labelled queries to score")
    missing = [query_id for query_id in qrels if query_id not in queries]
    if missing:
        raise ValueError(f"labelled query {missing[0]} has no text")

    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    for query_id, grades in qrels.items():
        hits = retriever.search(queries[query_id], k)
        ranked = [hit.id for hit in hits]
        ndcgs.append(ndcg_at_k(ranked, grades, k))
        recalls.append(recall_at_k(ranked, grades, k))
        mrrs.append(mrr_at_k(ranked, grades, k))

    ndcg = mean(ndcgs)
    # The floor is an nDCG@10 gate. Other cutoffs still report a scorecard.
    passed = k != EVAL_K or ndcg + 1e-12 >= NDCG_AT_10_FLOOR
    return Scorecard(
        dataset=dataset,
        k=k,
        query_count=len(qrels),
        ndcg_at_k=ndcg,
        recall_at_k=mean(recalls),
        mrr_at_k=mean(mrrs),
        passed=passed,
    )


def write_scorecard(scorecard: Scorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scorecard.model_dump_json(indent=2) + "\n", encoding="utf-8")
