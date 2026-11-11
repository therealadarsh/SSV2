import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssv2.loaders.jsonl import load_jsonl
from ssv2.loaders.scifact import (
    SCIFACT_CORPUS_FILE,
    SCIFACT_DATASET,
    SCIFACT_LICENSE,
    SCIFACT_QRELS_DATASET,
    SCIFACT_QRELS_FILE,
    SCIFACT_QRELS_REVISION,
    SCIFACT_QUERIES_FILE,
    SCIFACT_REVISION,
    download_scifact_corpus,
    download_scifact_qrels,
    download_scifact_queries,
    load_qrels_tsv,
    load_scifact_corpus,
    load_scifact_queries,
)


def test_builtin_corpus_is_four_mit_documents() -> None:
    path = Path(__file__).resolve().parents[1] / "assets" / "builtin.jsonl"
    documents = list(load_jsonl(path))
    assert [document.id for document in documents] == ["overview", "health", "scifact", "model"]
    assert {document.license for document in documents} == {"MIT"}


def test_jsonl_loader_skips_blank_lines_and_keeps_document_fields(tmp_path: Path) -> None:
    document = {
        "id": "a",
        "title": "Hello",
        "text": "World",
        "source": "notes",
        "url": None,
        "date": None,
        "license": "MIT",
        "metadata": {"topic": "greeting"},
    }
    path = tmp_path / "docs.jsonl"
    path.write_text("\n" + json.dumps(document) + "\n\n", encoding="utf-8")
    loaded = list(load_jsonl(path))
    assert len(loaded) == 1
    assert loaded[0].id == "a"
    assert loaded[0].metadata == {"topic": "greeting"}
    assert loaded[0].license == "MIT"


def test_jsonl_loader_reports_the_line_number_of_a_bad_record(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"id": "ok", "title": "t", "text": "b", "source": "s", "license": "MIT"}\n{}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"docs.jsonl:2"):
        list(load_jsonl(path))


def test_scifact_loader_records_the_noncommercial_license(tmp_path: Path) -> None:
    table = pa.table(
        {
            "_id": ["4983", "99"],
            "title": ["A paper", ""],
            "text": ["The abstract.", "Body only."],
        }
    )
    path = tmp_path / "corpus.parquet"
    pq.write_table(table, path)
    documents = list(load_scifact_corpus(path))
    assert [document.id for document in documents] == ["4983", "99"]
    assert documents[0].title == "A paper"
    assert documents[0].text == "The abstract."
    assert documents[0].source == "scifact"
    assert documents[0].license == SCIFACT_LICENSE
    assert "NC" in SCIFACT_LICENSE
    assert documents[1].title == ""


def test_qrels_skip_blanks_keep_the_higher_grade_and_reject_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "test.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n\nq1\td1\t0\nq1\td2\t2\nq2\td9\t0\n",
        encoding="utf-8",
    )
    assert load_qrels_tsv(path) == {"q1": {"d1": 1, "d2": 2}, "q2": {"d9": 0}}

    empty = tmp_path / "empty.tsv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_qrels_tsv(empty)

    bad = tmp_path / "bad.tsv"
    bad.write_text("query-id\tcorpus-id\tscore\nq\td\tnope\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.tsv:2"):
        load_qrels_tsv(bad)

    negative = tmp_path / "neg.tsv"
    negative.write_text("query-id\tcorpus-id\tscore\nq\td\t-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=">= 0"):
        load_qrels_tsv(negative)

    labels_only = tmp_path / "none.tsv"
    labels_only.write_text("query-id\tcorpus-id\tscore\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no labelled"):
        load_qrels_tsv(labels_only)


def test_query_parquet_rejects_a_blank_and_a_duplicate(tmp_path: Path) -> None:
    blank = tmp_path / "blank.parquet"
    pq.write_table(pa.table({"_id": ["q"], "text": ["  "]}), blank)
    with pytest.raises(ValueError, match="no text"):
        load_scifact_queries(blank)
    duplicate = tmp_path / "dup.parquet"
    pq.write_table(pa.table({"_id": ["q", "q"], "text": ["one", "two"]}), duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_scifact_queries(duplicate)


def test_scifact_download_uses_the_pinned_dataset_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return "/cache/corpus.parquet"

    monkeypatch.setattr("ssv2.loaders.scifact.hf_hub_download", fake_download)
    assert download_scifact_corpus() == Path("/cache/corpus.parquet")
    assert captured["repo_id"] == SCIFACT_DATASET == "BeIR/scifact"
    assert captured["filename"] == SCIFACT_CORPUS_FILE
    assert captured["repo_type"] == "dataset"
    assert captured["revision"] == SCIFACT_REVISION
    assert len(str(captured["revision"])) == 40

    captured.clear()
    monkeypatch.setattr("ssv2.loaders.scifact.hf_hub_download", fake_download)
    download_scifact_queries()
    assert captured["filename"] == SCIFACT_QUERIES_FILE
    assert captured["revision"] == SCIFACT_REVISION

    captured.clear()
    download_scifact_qrels()
    assert captured["repo_id"] == SCIFACT_QRELS_DATASET
    assert captured["filename"] == SCIFACT_QRELS_FILE
    assert captured["revision"] == SCIFACT_QRELS_REVISION
    assert len(SCIFACT_QRELS_REVISION) == 40
