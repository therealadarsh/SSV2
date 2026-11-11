"""FastAPI application.

Run locally with `make run`. In the container:
    uvicorn ssv2.api.app:create_app --factory

`create_app` is a factory instead of a module-level `app` so that importing this module has
no side effects, and each test can build an app with its own settings.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status

from ssv2 import __version__
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


def _require_ready(request: Request) -> None:
    if not request.app.state.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="not ready")


def _load_model(settings: Settings) -> Embedder:
    from ssv2.embedding import SentenceTransformersEmbedder

    return SentenceTransformersEmbedder(settings)
