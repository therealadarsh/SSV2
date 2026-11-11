"""The one document shape every loader produces.

Nothing downstream of a loader knows which dataset a row came from. The fields are the ones
in the plan: identity, the text we embed and show, where it came from, and the license we
are allowed to keep it under.
"""

import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    title: str = ""
    text: str = ""
    source: str = Field(min_length=1)
    url: str | None = None
    # The annotation cannot say `date | None`: the field name would shadow the type.
    date: datetime.date | None = None
    license: str = Field(min_length=1)
    # String values only, so filters are exact equality with no type surprises.
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "source", "license")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @model_validator(mode="after")
    def _has_title_or_text(self) -> "Document":
        if not self.title.strip() and not self.text.strip():
            raise ValueError("document needs a title or text")
        return self


def passage_text(document: Document) -> str:
    """The string that gets embedded.

    BEIR and MTEB concatenate the title and the body with a single space. Matching that is
    what makes a later SciFact score comparable to the published one.
    """
    title = document.title.strip()
    text = document.text.strip()
    if title and text:
        return f"{title} {text}"
    return title or text
