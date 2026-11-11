"""FastAPI application.

Run locally with `make run`. In the container:
    uvicorn ssv2.api.app:create_app --factory

`create_app` is a factory instead of a module-level `app` so that importing this module has
no side effects, and each test can build an app with its own settings.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status

from ssv2 import __version__
from ssv2.api.schemas import HealthResponse, ReadyResponse
from ssv2.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup runs to completion before uvicorn accepts any connection. Phase 1 loads the
        # embedding model and the index here and checks the index manifest against
        # `settings`; any failure raises, so the process exits instead of serving bad results.
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

    return app
