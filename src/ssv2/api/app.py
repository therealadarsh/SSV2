"""FastAPI application.

Run locally with `make run`. In the container:
    uvicorn ssv2.api.app:create_app --factory

`create_app` is a factory instead of a module-level `app` so that importing this module has
no side effects, and each test can build an app with its own settings.
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse

from ssv2 import __version__
from ssv2.api.page import SEARCH_PAGE
from ssv2.api.schemas import (
    HealthResponse,
    InfoResponse,
    ReadyResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from ssv2.config import Settings, get_settings
from ssv2.embedding import Embedder
from ssv2.retrieval import DenseRetriever, Retriever

_access_log = logging.getLogger("ssv2.access")


def create_app(
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the app.

    Pass `retriever` in tests that are not about the index. Pass `embedder` to load
    `settings.index_dir` with a stand-in model. With neither, startup loads the configured
    model and refuses an index whose manifest does not match it.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup runs to completion before uvicorn accepts any connection. A manifest
        # mismatch raises here, so the process exits instead of serving bad results.
        if retriever is not None:
            app.state.retriever = retriever
        else:
            model = embedder if embedder is not None else _load_model(settings)
            app.state.retriever = DenseRetriever(settings.index_dir, model, settings)
        app.state.ready = True
        yield
        # Shutting down: report not-ready so a load balancer stops sending new requests while
        # in-flight ones finish.
        app.state.ready = False

    app = FastAPI(title="SSV2", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.ready = False
    app.middleware("http")(_access_log_middleware)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> HTMLResponse:
        """Minimal search page. It calls POST /v1/search and renders the hits."""
        return HTMLResponse(SEARCH_PAGE)

    @app.get("/health", tags=["ops"])
    def health() -> HealthResponse:
        """Liveness: the process is up and answering. Never touches the model or index, so
        it stays cheap and doesn't fail because of a dependency."""
        return HealthResponse(status="ok", version=__version__)

    @app.get(
        "/ready",
        tags=["ops"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyResponse}},
    )
    def ready(request: Request, response: Response) -> ReadyResponse:
        """Readiness: everything needed to answer searches is loaded. Returns 503 otherwise,
        which tells an orchestrator to hold traffic back without restarting the process."""
        if request.app.state.ready:
            return ReadyResponse(status="ready")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready")

    @app.get("/v1/info", tags=["search"])
    def info(request: Request) -> InfoResponse:
        """The index this process is serving: dataset, size, model, revision, build time."""
        _require_ready(request)
        manifest = request.app.state.retriever.info
        return InfoResponse(
            dataset=manifest.dataset,
            document_count=manifest.document_count,
            embedding_model=manifest.embedding_model,
            embedding_model_revision=manifest.embedding_model_revision,
            built_at=manifest.built_at,
        )

    @app.post(
        "/v1/search",
        tags=["search"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Index not loaded"}},
    )
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        """Nearest documents by cosine similarity. `filters` keeps hits whose metadata matches."""
        _require_ready(request)
        hits = request.app.state.retriever.search(body.query, body.k, body.filters)
        return SearchResponse(hits=[SearchHit.model_validate(hit.model_dump()) for hit in hits])

    return app


async def _access_log_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """One JSON line per request: method, path, status, and how long it took.

    The query text stays out of the log. Duration is on the response as
    ``x-request-duration-ms`` so the search page can show it.
    """
    started = time.perf_counter()
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    response: Response | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        _access_log.info(
            json.dumps(
                {
                    "event": "request",
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 3),
                }
            )
        )
    if response is None:
        raise RuntimeError("request produced no response")
    response.headers["x-request-duration-ms"] = f"{duration_ms:.3f}"
    return response


def _require_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="not ready")


def _load_model(settings: Settings) -> Embedder:
    from ssv2.embedding import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(settings)
