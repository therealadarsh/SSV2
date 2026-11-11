"""Response models. They define the API contract and appear in the OpenAPI docs at /docs."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
