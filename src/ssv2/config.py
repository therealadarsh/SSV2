"""All configuration, read from environment variables in one place.

Every setting has a default that works locally and can be overridden with an `SSV2_`-prefixed
environment variable (for example `SSV2_INDEX_DIR=/data/index`). The Docker image, CI and
tests configure the service this way, without code changes.
"""

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SSV2_", frozen=True)

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Hugging Face model id used to embed documents and queries.",
    )
    embedding_model_revision: str = Field(
        default="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        description="Full commit SHA of the model on Hugging Face.",
    )
    index_dir: Path = Field(
        default=Path("indexes/scifact"),
        description="Directory holding the built index (vectors, documents, manifest).",
    )

    @field_validator("embedding_model_revision")
    @classmethod
    def _revision_is_commit_sha(cls, value: str) -> str:
        # A branch or tag name such as "main" can point at different weights tomorrow, which
        # would silently change every score. Only an immutable commit SHA is reproducible.
        if not _COMMIT_SHA.fullmatch(value):
            raise ValueError(
                f"must be a full 40-character commit SHA, not a branch or tag name (got {value!r})"
            )
        return value


@lru_cache
def get_settings() -> Settings:
    """Settings for this process, read from the environment once."""
    return Settings()
