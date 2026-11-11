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
