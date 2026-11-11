import datetime
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ssv2.cli import main
from ssv2.documents import passage_text
from ssv2.loaders.arxiv import (
    ARXIV_DATASET,
    ARXIV_LICENSE,
    ARXIV_REVISION,
    ARXIV_SHARD,
    download_arxiv_shard,
    load_arxiv_shard,
)
from tests.fakes import HashEmbedder


def _write(path: Path) -> None:
    table = pa.table(
        {
            "id": ["keep", "nc", "math", "blank", "keep", "early", "slash"],
            "title": ["Kept", "Nope", "Physics", "   ", "Duplicate", "Early", "Slash"],
            "abstract": ["language", "language", "fields", "   ", "again", "old work", "also cs"],
            "categories": ["cs.CL cs.AI", "cs.CL", "math.ST", "cs.CL", "cs.CL", "cs.CL", "cs.LG"],
            "license": [
                "http://creativecommons.org/licenses/by/4.0/",
                "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "http://creativecommons.org/licenses/by/4.0/",
                "http://creativecommons.org/licenses/by/4.0/",
                "http://creativecommons.org/licenses/by/4.0/",
                "http://creativecommons.org/licenses/by/4.0/",
                "http://creativecommons.org/licenses/by/4.0",
            ],
            "update_date": [
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2024, 5, 2),
                datetime.datetime(2019, 1, 1),
                datetime.datetime(2024, 7, 1),
            ],
        }
    )
    pq.write_table(table, path)


def test_loader_keeps_only_cc_by_cs_papers_inside_the_window(tmp_path: Path) -> None:
    path = tmp_path / "shard.parquet"
    _write(path)
    documents = list(
        load_arxiv_shard(
            path, date_from=datetime.date(2024, 1, 1), date_to=datetime.date(2024, 12, 31)
        )
    )
    assert [document.id for document in documents] == ["keep", "slash"]
    assert documents[0].license == ARXIV_LICENSE
    assert documents[0].source == "arxiv"
    assert documents[0].url == "https://arxiv.org/abs/keep"
    assert documents[0].date == datetime.date(2024, 5, 1)
    assert documents[0].metadata["category"] == "cs.CL"
    assert documents[0].metadata["categories"] == "cs.CL cs.AI"
    assert "by/4.0" in documents[0].metadata["license_url"]
    assert "NC" not in documents[0].license


def test_category_prefix_limit_and_a_reversed_window(tmp_path: Path) -> None:
    path = tmp_path / "shard.parquet"
    _write(path)
    only_lg = list(load_arxiv_shard(path, category_prefix="cs.LG"))
    assert [document.id for document in only_lg] == ["slash"]
    limited = list(load_arxiv_shard(path, limit=1))
    assert [document.id for document in limited] == ["keep"]
    with pytest.raises(ValueError, match="date_from"):
        list(
            load_arxiv_shard(
                path, date_from=datetime.date(2025, 1, 1), date_to=datetime.date(2024, 1, 1)
            )
        )
    with pytest.raises(ValueError, match="limit"):
        list(load_arxiv_shard(path, limit=0))


def test_arxiv_download_uses_the_pinned_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return "/cache/shard.parquet"

    monkeypatch.setattr("ssv2.loaders.arxiv.hf_hub_download", fake_download)
    assert download_arxiv_shard() == Path("/cache/shard.parquet")
    assert captured["repo_id"] == ARXIV_DATASET
    assert captured["filename"] == ARXIV_SHARD
    assert captured["revision"] == ARXIV_REVISION
    assert captured["repo_type"] == "dataset"
    assert len(ARXIV_REVISION) == 40


def test_index_arxiv_then_hybrid_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL", "fake/hash")
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL_REVISION", "a" * 40)
    shard = tmp_path / "shard.parquet"
    _write(shard)
    index_dir = tmp_path / "index"
    code = main(
        ["index", "arxiv", "--path", str(shard), "--limit", "10", "--output", str(index_dir)],
        embedder_factory=lambda _settings: HashEmbedder(),
    )
    assert code == 0
    assert "arxiv" in capsys.readouterr().out
    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table({"_id": ["q1"], "text": [passage_text_of_keep(shard)]}),
        queries,
    )
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\tkeep\t1\n", encoding="utf-8")
    code = main(
        [
            "eval",
            "--index",
            str(index_dir),
            "--queries",
            str(queries),
            "--qrels",
            str(qrels),
            "--retriever",
            "hybrid",
            "--output",
            str(tmp_path / "scorecard.json"),
        ],
        embedder_factory=lambda _settings: HashEmbedder(),
    )
    assert code == 0
    card = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert card["ndcg_at_k"] == 1.0


def passage_text_of_keep(path: Path) -> str:
    document = next(load_arxiv_shard(path, limit=1))
    assert document.id == "keep"
    return passage_text(document)
