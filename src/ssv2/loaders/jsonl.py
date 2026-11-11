"""Load a corpus that is already in the document schema, one JSON object per line."""

from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from ssv2.documents import Document


def load_jsonl(path: Path) -> Iterator[Document]:
    """Yield documents from `path`. Blank lines are ignored.

    A bad line raises `ValueError` naming the file and the 1-based line number.
    Metadata values must be strings; numbers and nested objects are rejected.
    """
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield Document.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
