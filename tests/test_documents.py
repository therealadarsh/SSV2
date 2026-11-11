import datetime

import pytest
from pydantic import ValidationError

from ssv2.documents import Document, passage_text


def test_passage_joins_title_and_text_with_a_space() -> None:
    document = Document(
        id="1",
        title="Vitamin D",
        text="Low levels correlate with risk.",
        source="fixture",
        license="MIT",
    )
    assert passage_text(document) == "Vitamin D Low levels correlate with risk."


def test_passage_uses_whichever_field_is_present() -> None:
    untitled = Document(
        id="1", title="", text="Only the abstract.", source="fixture", license="MIT"
    )
    assert passage_text(untitled) == "Only the abstract."
    bodiless = Document(id="2", title="Only a title", text="  ", source="fixture", license="MIT")
    assert passage_text(bodiless) == "Only a title"


def test_document_requires_an_id_source_license_and_some_text() -> None:
    with pytest.raises(ValidationError):
        Document(id="", title="t", text="b", source="s", license="MIT")
    with pytest.raises(ValidationError):
        Document(id="1", title="", text="  ", source="s", license="MIT")
    with pytest.raises(ValidationError):
        Document(id="1", title="t", text="b", source="", license="MIT")
    with pytest.raises(ValidationError):
        Document(id="1", title="t", text="b", source="s", license="")


def test_document_round_trips_through_json_including_the_date() -> None:
    document = Document(
        id="1",
        title="t",
        text="b",
        source="arxiv",
        url="https://example.test/1",
        date=datetime.date(2024, 5, 6),
        license="CC BY 4.0",
        metadata={"category": "cs.CL"},
    )
    assert Document.model_validate_json(document.model_dump_json()) == document


def test_metadata_values_are_strings() -> None:
    with pytest.raises(ValidationError):
        Document.model_validate(
            {
                "id": "1",
                "title": "t",
                "text": "b",
                "source": "s",
                "license": "MIT",
                "metadata": {"year": 2024},
            }
        )
