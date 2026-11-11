"""Unit tests for the HTTP layer. No model, no index, no network: they run in milliseconds."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ssv2 import __version__
from ssv2.api.app import create_app
from ssv2.config import Settings


@pytest.fixture
def app() -> FastAPI:
    return create_app(Settings())


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


def test_openapi_documents_both_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ready"} <= paths.keys()
