"""arXiv metadata loader. Only CC BY 4.0 papers are kept.

The metadata snapshot is CC0. Each paper's text is under the license on that row, and
this loader drops every row whose license is not CC BY 4.0. Computer-science papers are
the default slice: a category token must start with ``cs.``.
"""

import datetime
from collections.abc import Iterator, Mapping
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ssv2.documents import Document

ARXIV_DATASET = "librarian-bots/arxiv-metadata-snapshot"
ARXIV_REVISION = "c428226b689a0e17db2cd55265133b9eca15d97f"
# One shard, not the 2.7 GB snapshot. Shard 0 is the recent papers, where CC BY 4.0 is common.
ARXIV_SHARD = "data/train-00000-of-00011.parquet"
CC_BY_4_URL = "http://creativecommons.org/licenses/by/4.0/"
ARXIV_LICENSE = "CC BY 4.0"


def download_arxiv_shard() -> Path:
    """Download the pinned shard. The file is metadata; papers are filtered after."""
    path = hf_hub_download(
        repo_id=ARXIV_DATASET,
        filename=ARXIV_SHARD,
        repo_type="dataset",
        revision=ARXIV_REVISION,
    )
    return Path(path)


def load_arxiv_shard(
    path: Path,
    *,
    category_prefix: str = "cs.",
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
    limit: int | None = None,
) -> Iterator[Document]:
    """Yield CC BY 4.0 papers from one metadata parquet file."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("date_from is after date_to")
    table = pq.read_table(
        path, columns=["id", "title", "abstract", "categories", "license", "update_date"]
    )
    produced = 0
    seen: set[str] = set()
    for row in table.to_pylist():
        document = _document(
            row, category_prefix=category_prefix, date_from=date_from, date_to=date_to
        )
        if document is None or document.id in seen:
            continue
        seen.add(document.id)
        produced += 1
        yield document
        if limit is not None and produced >= limit:
            return


def _document(
    row: Mapping[str, object],
    *,
    category_prefix: str,
    date_from: datetime.date | None,
    date_to: datetime.date | None,
) -> Document | None:
    if not _is_cc_by_4(row.get("license")):
        return None
    categories = _categories(row.get("categories"))
    if category_prefix and not any(token.startswith(category_prefix) for token in categories):
        return None
    title = _clean(row.get("title"))
    text = _clean(row.get("abstract"))
    if not title and not text:
        return None
    paper_id = _clean(row.get("id"))
    if not paper_id:
        return None
    updated = _as_date(row.get("update_date"))
    if date_from is not None and (updated is None or updated < date_from):
        return None
    if date_to is not None and (updated is None or updated > date_to):
        return None
    primary = categories[0] if categories else ""
    return Document(
        id=paper_id,
        title=title,
        text=text,
        source="arxiv",
        url=f"https://arxiv.org/abs/{paper_id}",
        date=updated,
        license=ARXIV_LICENSE,
        metadata={
            "category": primary,
            "categories": " ".join(categories),
            "license_url": CC_BY_4_URL,
            "date": updated.isoformat() if updated is not None else "",
        },
    )


def _is_cc_by_4(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().rstrip("/") == CC_BY_4_URL.rstrip("/")


def _categories(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [token for token in value.split() if token]


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _as_date(value: object) -> datetime.date | None:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None
