import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssv2.cli import main
from ssv2.documents import Document, passage_text
from ssv2.evals.metrics import ndcg_at_k
from ssv2.evals.run import NDCG_AT_10_FLOOR, score_retrieval
from ssv2.index import build_index
from ssv2.retrieval import Hit
from tests.fakes import FAKE_REVISION, HashEmbedder


class ScriptedRetriever:
    def __init__(self, hits_by_query: dict[str, list[str]]) -> None:
        self._hits = hits_by_query

    def search(self, query: str, k: int, filters: Mapping[str, str] | None = None) -> list[Hit]:
        del filters
        return [
            Hit(id=doc_id, title=doc_id, text=doc_id, score=1.0, metadata={})
            for doc_id in self._hits[query][:k]
        ]


def test_scorecard_averages_labelled_queries_and_ignores_unlabelled_ones() -> None:
    retriever = ScriptedRetriever({"finds it": ["a"], "misses": ["z"]})
    card = score_retrieval(
        retriever,
        queries={"finds it": "finds it", "misses": "misses", "unused": "unused"},
        qrels={"finds it": {"a": 1}, "misses": {"a": 1}},
        dataset="toy",
        k=10,
    )
    assert card.query_count == 2
    assert card.ndcg_at_k == pytest.approx(0.5)
    assert card.recall_at_k == pytest.approx(0.5)
    assert card.mrr_at_k == pytest.approx(0.5)
    assert card.passed is False
    assert card.ndcg_at_10_floor == NDCG_AT_10_FLOOR
    assert card.scale == "0-1"


def test_scorecard_passes_when_ndcg_is_above_the_floor() -> None:
    retriever = ScriptedRetriever({"q": ["a"]})
    card = score_retrieval(
        retriever, queries={"q": "q"}, qrels={"q": {"a": 1}}, dataset="toy", k=10
    )
    assert card.ndcg_at_k == 1.0
    assert card.passed is True


def test_a_cutoff_other_than_10_does_not_use_the_ndcg_at_10_floor() -> None:
    retriever = ScriptedRetriever({"q": ["miss"]})
    card = score_retrieval(retriever, queries={"q": "q"}, qrels={"q": {"a": 1}}, dataset="toy", k=1)
    assert card.ndcg_at_k == 0.0
    assert card.passed is True


def test_missing_query_text_and_empty_labels_are_rejected() -> None:
    retriever = ScriptedRetriever({})
    with pytest.raises(ValueError, match="no text"):
        score_retrieval(retriever, queries={}, qrels={"q": {"a": 1}}, dataset="toy")
    with pytest.raises(ValueError, match="no labelled"):
        score_retrieval(retriever, queries={"q": "q"}, qrels={}, dataset="toy")
    with pytest.raises(ValueError, match="at least 1"):
        score_retrieval(retriever, queries={"q": "q"}, qrels={"q": {"a": 1}}, dataset="toy", k=0)


def test_metric_used_by_the_scorecard_matches_a_single_query() -> None:
    # Guards the scorecard against drifting from the hand-worked nDCG definition.
    assert ndcg_at_k(["a"], {"a": 1}, 10) == 1.0


def _write_queries(path: Path, rows: list[tuple[str, str]]) -> None:
    table = pa.table({"_id": [row[0] for row in rows], "text": [row[1] for row in rows]})
    pq.write_table(table, path)


def _factory(settings: object) -> HashEmbedder:
    del settings
    return HashEmbedder()


def test_eval_command_round_trip_on_a_tiny_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL", "fake/hash")
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL_REVISION", FAKE_REVISION)
    document = Document(id="alpha", title="Alpha", text="apples", source="notes", license="MIT")
    index_dir = tmp_path / "index"
    build_index([document], HashEmbedder(), index_dir, dataset="notes")
    queries = tmp_path / "queries.parquet"
    _write_queries(queries, [("q1", passage_text(document))])
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\talpha\t1\n", encoding="utf-8")
    output = tmp_path / "scorecard.json"
    code = main(
        [
            "eval",
            "--index",
            str(index_dir),
            "--queries",
            str(queries),
            "--qrels",
            str(qrels),
            "--output",
            str(output),
        ],
        embedder_factory=_factory,
    )
    assert code == 0
    card = json.loads(output.read_text(encoding="utf-8"))
    assert card["ndcg_at_k"] == 1.0
    assert card["passed"] is True
    assert json.loads(capsys.readouterr().out)["query_count"] == 1


def test_eval_command_fails_when_nothing_relevant_is_retrieved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL", "fake/hash")
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL_REVISION", FAKE_REVISION)
    document = Document(id="alpha", title="Alpha", text="apples", source="notes", license="MIT")
    index_dir = tmp_path / "index"
    build_index([document], HashEmbedder(), index_dir, dataset="notes")
    queries = tmp_path / "queries.parquet"
    _write_queries(queries, [("q1", "apples")])
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\tmissing\t1\n", encoding="utf-8")
    code = main(
        ["eval", "--index", str(index_dir), "--queries", str(queries), "--qrels", str(qrels)],
        embedder_factory=_factory,
    )
    assert code == 1
    assert "floor" in capsys.readouterr().err.lower()
