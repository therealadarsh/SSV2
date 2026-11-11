"""Request and response models. They define the API contract and appear in the OpenAPI docs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ssv2.retrieval import MAX_K, MIN_K


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=10, ge=MIN_K, le=MAX_K)
    filters: dict[str, str] | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class SearchHit(BaseModel):
    id: str
    title: str
    text: str
    score: float
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class InfoResponse(BaseModel):
    dataset: str
    document_count: int
    embedding_model: str
    embedding_model_revision: str
    built_at: datetime
