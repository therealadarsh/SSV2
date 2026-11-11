import json
from pathlib import Path

import pytest

from ssv2.cli import main
from ssv2.config import get_settings
from ssv2.documents import Document, passage_text
from tests.fakes import HashEmbedder


def _factory(settings: object) -> HashEmbedder:
    del settings
    return HashEmbedder()


def test_index_then_search_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL", "fake/hash")
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL_REVISION", "a" * 40)
    corpus = tmp_path / "corpus.jsonl"
    documents = [
        Document(
            id="alpha", title="Alpha", text="apples and orchards", source="notes", license="MIT"
        ),
        Document(id="beta", title="Beta", text="bridges and rivers", source="notes", license="MIT"),
    ]
    payload = "".join(document.model_dump_json() + "\n" for document in documents)
    corpus.write_text(payload, encoding="utf-8")
    output = tmp_path / "index"
    code = main(
        ["index", "jsonl", str(corpus), "--dataset", "notes", "--output", str(output)],
        embedder_factory=_factory,
    )
    assert code == 0
    assert "2" in capsys.readouterr().out

    code = main(
        ["search", passage_text(documents[1]), "-k", "1", "--index", str(output), "--json"],
        embedder_factory=_factory,
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"][0]["id"] == "beta"
    assert payload["hits"][0]["score"] > 0.9


def test_search_without_an_index_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSV2_INDEX_DIR", str(tmp_path / "absent"))
    get_settings.cache_clear()
    code = main(["search", "hello"], embedder_factory=_factory)
    assert code == 1
    assert "index" in capsys.readouterr().err.lower()


def test_k_out_of_range_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["search", "hello", "-k", "0"], embedder_factory=_factory)
    assert code == 2
    assert "k" in capsys.readouterr().err.lower()


def test_bench_rejects_bad_arguments_before_loading_an_index(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["bench", "-n", "0"], embedder_factory=_factory) == 2
    assert "n" in capsys.readouterr().err.lower()
    assert main(["bench", "--query", "  "], embedder_factory=_factory) == 2
    assert main(["bench", "-k", "0"], embedder_factory=_factory) == 2
    assert main(["bench", "--max-p95-ms", "0"], embedder_factory=_factory) == 2


def test_bench_reports_percentiles_and_can_fail_the_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL", "fake/hash")
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL_REVISION", "a" * 40)
    corpus = tmp_path / "corpus.jsonl"
    document = Document(id="alpha", title="Alpha", text="apples", source="notes", license="MIT")
    corpus.write_text(document.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "index"
    assert (
        main(
            ["index", "jsonl", str(corpus), "--dataset", "notes", "--output", str(output)],
            embedder_factory=_factory,
        )
        == 0
    )
    capsys.readouterr()
    code = main(
        ["bench", "--index", str(output), "-n", "5", "--max-p95-ms", "10000"],
        embedder_factory=_factory,
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["n"] == 5
    assert report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"] <= report["max_ms"]
    code = main(
        ["bench", "--index", str(output), "-n", "3", "--max-p95-ms", "0.000001"],
        embedder_factory=_factory,
    )
    assert code == 1
    assert "p95" in capsys.readouterr().err.lower()
