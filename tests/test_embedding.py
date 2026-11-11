import os
from pathlib import Path

import numpy as np
import pytest

from ssv2.embedding import QUERY_INSTRUCTION, format_query, l2_normalize, prepare_hf_home


def test_query_instruction_matches_the_bge_model_card() -> None:
    # Trailing space is part of the published instruction, so the query is not glued to the colon.
    assert QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "
    assert format_query("does vitamin D help?") == QUERY_INSTRUCTION + "does vitamin D help?"


def test_l2_normalize_makes_rows_unit_length_and_keeps_direction() -> None:
    vectors = np.array([[3.0, 0.0], [0.0, -4.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    assert normalized.dtype == np.float32
    assert np.allclose(norms, [1.0, 1.0])
    assert normalized[0, 0] > 0
    assert normalized[1, 1] < 0


def test_l2_normalize_accepts_a_single_vector() -> None:
    normalized = l2_normalize(np.array([0.0, 5.0], dtype=np.float32))
    assert normalized.shape == (2,)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_readonly_huggingface_cache_is_copied_somewhere_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "hf"
    (cache / "hub").mkdir(parents=True)
    (cache / "hub" / "marker.txt").write_text("weights", encoding="utf-8")
    (cache / "hub" / "marker.txt").chmod(0o444)
    (cache / "hub").chmod(0o555)
    cache.chmod(0o555)
    monkeypatch.setenv("HF_HOME", str(cache))
    try:
        prepared = prepare_hf_home()
        assert prepared != cache
        assert os.access(prepared, os.W_OK)
        assert (prepared / "hub" / "marker.txt").read_text(encoding="utf-8") == "weights"
        assert os.environ["HF_HOME"] == str(prepared)
    finally:
        cache.chmod(0o755)
        (cache / "hub").chmod(0o755)


def test_writable_huggingface_cache_is_left_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "hf"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    assert prepare_hf_home() == cache
