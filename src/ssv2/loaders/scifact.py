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


def download_scifact_corpus() -> Path:
    """Download the pinned corpus file and return the local path. Needs the network."""
    downloaded = hf_hub_download(
        repo_id=SCIFACT_DATASET,
        filename=SCIFACT_CORPUS_FILE,
        repo_type="dataset",
        revision=SCIFACT_REVISION,
    )
    return Path(downloaded)
