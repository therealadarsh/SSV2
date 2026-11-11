"""SciFact corpus loader.

SciFact is the evaluation benchmark (5,183 abstracts, expert relevance labels). Upstream
SciFact is CC BY-NC 2.0, so it is non-commercial. The BEIR copy on Hugging Face is tagged
CC BY-SA 4.0; this project still treats the corpus as non-commercial and uses it for
evaluation, not as a corpus we redistribute. The image therefore does not contain it.

The download is pinned to a Hugging Face commit so a later dataset edit cannot change
scores without an explicit revision bump.
"""

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ssv2.documents import Document

SCIFACT_DATASET = "BeIR/scifact"
# Commit of the dataset repo. Corpus split: 5,183 rows, columns _id, title, text.
SCIFACT_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
SCIFACT_CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
SCIFACT_QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
# Relevance labels live in a separate dataset repo. `test.tsv` is the official test split.
SCIFACT_QRELS_DATASET = "BeIR/scifact-qrels"
SCIFACT_QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
SCIFACT_QRELS_FILE = "test.tsv"
# The license we record on every row. Non-commercial, per the plan, not the HF tag.
SCIFACT_LICENSE = "CC BY-NC 2.0"


def load_scifact_corpus(path: Path) -> Iterator[Document]:
    """Yield documents from a BEIR SciFact corpus parquet file."""
    table = pq.read_table(path, columns=["_id", "title", "text"])
    for row in table.to_pylist():
        title = row["title"] or ""
        text = row["text"] or ""
        yield Document(
            id=str(row["_id"]),
            title=title,
            text=text,
            source="scifact",
            url=None,
            date=None,
            license=SCIFACT_LICENSE,
            metadata={},
        )


def load_scifact_queries(path: Path) -> dict[str, str]:
    """Map query id to query text. Train and test queries are both in this file."""
    table = pq.read_table(path, columns=["_id", "text"])
    queries: dict[str, str] = {}
    for row in table.to_pylist():
        query_id = str(row["_id"])
        text = (row["text"] or "").strip()
        if not text:
            raise ValueError(f"query {query_id} has no text")
        if query_id in queries:
            raise ValueError(f"duplicate query id {query_id}")
        queries[query_id] = text
    if not queries:
        raise ValueError(f"{path} has no queries")
    return queries


def load_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    """Map query id to {document id: grade}.

    The header must be ``query-id``, ``corpus-id``, ``score``. A repeated pair keeps the
    higher grade. Grade 0 is stored and means "judged not relevant".
    """
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if header.strip() == "":
            raise ValueError(f"{path.name}: missing header")
        columns = header.rstrip("\n").split("\t")
        if columns != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"{path.name}: expected header query-id, corpus-id, score")
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path.name}:{line_number}: expected 3 columns")
            query_id, doc_id, raw_grade = parts
            if not query_id or not doc_id:
                raise ValueError(f"{path.name}:{line_number}: blank id")
            try:
                grade = int(raw_grade)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: score must be an integer") from exc
            if grade < 0:
                raise ValueError(f"{path.name}:{line_number}: score must be >= 0")
            grades = qrels.setdefault(query_id, {})
            grades[doc_id] = max(grades.get(doc_id, 0), grade)
    if not qrels:
        raise ValueError(f"{path.name}: no labelled queries")
    return qrels


def download_scifact_queries() -> Path:
    return Path(
        hf_hub_download(
            repo_id=SCIFACT_DATASET,
            filename=SCIFACT_QUERIES_FILE,
            repo_type="dataset",
            revision=SCIFACT_REVISION,
        )
    )


def download_scifact_qrels() -> Path:
    return Path(
        hf_hub_download(
            repo_id=SCIFACT_QRELS_DATASET,
            filename=SCIFACT_QRELS_FILE,
            repo_type="dataset",
            revision=SCIFACT_QRELS_REVISION,
        )
    )


def download_scifact_corpus() -> Path:
    """Download the pinned corpus file and return the local path. Needs the network."""
    downloaded = hf_hub_download(
        repo_id=SCIFACT_DATASET,
        filename=SCIFACT_CORPUS_FILE,
        repo_type="dataset",
        revision=SCIFACT_REVISION,
    )
    return Path(downloaded)
