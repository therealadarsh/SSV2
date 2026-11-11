"""Unit tests for the HTTP layer. No model, no index, no network: they run in milliseconds."""

import datetime
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ssv2 import __version__
from ssv2.api.app import create_app
from ssv2.config import Settings
from ssv2.documents import Document, passage_text
from ssv2.index import IndexManifestError, build_index
from tests.fakes import HashEmbedder, StubRetriever


@pytest.fixture
def retriever() -> StubRetriever:
    return StubRetriever()


@pytest.fixture
def app(retriever: StubRetriever) -> FastAPI:
    return create_app(Settings(), retriever=retriever)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Using TestClient as a context manager runs the lifespan (startup and shutdown).
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_ok_and_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_ready_after_startup(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_not_ready(app: FastAPI, client: TestClient) -> None:
    app.state.ready = False
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def _is_ready(app: FastAPI) -> bool:
    return bool(app.state.ready)


def test_not_ready_before_startup_and_after_shutdown(app: FastAPI) -> None:
    assert not _is_ready(app)
    with TestClient(app):
        assert _is_ready(app)
    assert not _is_ready(app)


def test_settings_are_attached_to_the_app() -> None:
    settings = Settings(embedding_model="some/other-model")
    assert create_app(settings).state.settings is settings


def test_openapi_documents_the_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ready", "/v1/search", "/v1/info"} <= paths.keys()


def test_search_returns_hits(client: TestClient, retriever: StubRetriever) -> None:
    response = client.post(
        "/v1/search", json={"query": "vitamin D", "k": 3, "filters": {"topic": "demo"}}
    )
    assert response.status_code == 200
    assert response.json() == {
        "hits": [
            {
                "id": "doc-1",
                "title": "Stub title",
                "text": "Stub body",
                "score": 0.5,
                "metadata": {"topic": "demo"},
            }
        ]
    }
    assert retriever.calls == [("vitamin D", 3, {"topic": "demo"})]


def test_search_rejects_a_blank_query_and_a_k_out_of_range(client: TestClient) -> None:
    assert client.post("/v1/search", json={"query": "  "}).status_code == 422
    assert client.post("/v1/search", json={"query": "ok", "k": 0}).status_code == 422
    assert client.post("/v1/search", json={"query": "ok", "k": 101}).status_code == 422


def test_search_is_503_when_not_ready(app: FastAPI, client: TestClient) -> None:
    app.state.ready = False
    response = client.post("/v1/search", json={"query": "vitamin D"})
    assert response.status_code == 503


def test_info_reports_the_manifest(client: TestClient) -> None:
    response = client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == "stub"
    assert body["document_count"] == 1
    assert body["embedding_model"] == "fake/hash"
    assert body["embedding_model_revision"] == "a" * 40
    assert body["built_at"].startswith("2026-11-11")


def test_app_loads_an_index_and_searches_it(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    document = Document(
        id="only", title="Only", text="unique phrase", source="fixture", license="MIT"
    )
    build_index([document], embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    app = create_app(settings, embedder=embedder)
    with TestClient(app) as client:
        info = client.get("/v1/info")
        assert info.status_code == 200
        assert info.json()["dataset"] == "fixture"
        assert info.json()["document_count"] == 1
        found = client.post("/v1/search", json={"query": passage_text(document), "k": 1})
        assert found.status_code == 200
        assert found.json()["hits"][0]["id"] == "only"


def test_search_returns_422_for_a_bad_date_filter(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    document = Document(
        id="only",
        title="Only",
        text="unique phrase",
        source="fixture",
        license="MIT",
        date=datetime.date(2024, 1, 1),
        metadata={"date": "2024-01-01"},
    )
    build_index([document], embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model=embedder.model_id,
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    app = create_app(settings, embedder=embedder)
    with TestClient(app) as client:
        bad = client.post("/v1/search", json={"query": "unique", "filters": {"date_from": "nope"}})
        assert bad.status_code == 422
        kept = client.post(
            "/v1/search", json={"query": "unique", "filters": {"date_from": "2024-01-01"}}
        )
        assert kept.status_code == 200
        assert kept.json()["hits"][0]["id"] == "only"


def test_startup_refuses_an_index_from_another_model(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    document = Document(
        id="only", title="Only", text="unique phrase", source="fixture", license="MIT"
    )
    build_index([document], embedder, tmp_path, dataset="fixture")
    settings = Settings(
        embedding_model="other/model",
        embedding_model_revision=embedder.revision,
        index_dir=tmp_path,
    )
    app = create_app(settings, embedder=embedder)
    with pytest.raises(IndexManifestError, match="other/model"), TestClient(app):
        pass


def test_home_page_is_a_search_form(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    page = response.text
    assert 'id="ssv2-search"' in page
    assert "novalidate" in page
    assert 'id="query"' in page
    assert "/v1/search" in page
    assert "Enter a query." in page
    assert "No results." in page
    assert "textContent" in page
    assert "x-request-duration-ms" in page


def _access(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [json.loads(record.message) for record in caplog.records if record.name == "ssv2.access"]


def test_requests_are_logged_as_json_with_timing(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ssv2.access")
    response = client.get("/health")
    assert response.status_code == 200
    duration = float(response.headers["x-request-duration-ms"])
    assert duration >= 0
    health = _access(caplog)[-1]
    assert health == {
        "event": "request",
        "method": "GET",
        "path": "/health",
        "status": 200,
        "duration_ms": health["duration_ms"],
    }
    assert cast(float, health["duration_ms"]) >= 0

    found = client.post("/v1/search", json={"query": "vitamin D secret", "k": 1})
    assert found.status_code == 200
    logged = _access(caplog)
    assert "vitamin D secret" not in json.dumps(logged)
    assert logged[-1]["path"] == "/v1/search"
    assert logged[-1]["status"] == 200

    missing = client.get("/missing")
    assert missing.status_code == 404
    logged = _access(caplog)
    assert logged[-1]["path"] == "/missing"
    assert logged[-1]["status"] == 404
