from pathlib import Path

import pytest
from pydantic import ValidationError

from ssv2.config import Settings


def test_defaults_match_the_plan() -> None:
    settings = Settings()
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_model_revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert settings.embedding_model_dir is None


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSV2_INDEX_DIR", "/data/my-index")
    monkeypatch.setenv("SSV2_EMBEDDING_MODEL", "org/model")
    settings = Settings()
    assert settings.index_dir == Path("/data/my-index")
    assert settings.embedding_model == "org/model"


@pytest.mark.parametrize(
    "revision", ["main", "v1.5", "5c38ec7", "5C38EC7C405EC4B44B94CC5A9BB96E735B38267A"]
)
def test_revision_must_be_a_full_commit_sha(revision: str) -> None:
    with pytest.raises(ValidationError, match="40-character commit SHA"):
        Settings(embedding_model_revision=revision)


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.embedding_model = "changed"  # type: ignore[misc]
