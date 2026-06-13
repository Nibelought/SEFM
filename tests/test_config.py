"""Tests for the .env persistence helper used by the settings dialog."""

from __future__ import annotations

from pathlib import Path

from app.config import update_env_file


def test_update_env_file_upserts_and_preserves(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# user config\n"
        "SEFM_LLM_N_THREADS=6\n"
        "SEFM_ACCELERATION=auto\n",
        encoding="utf-8",
    )

    update_env_file(
        {"SEFM_ACCELERATION": "gpu", "SEFM_EMBEDDING_DEVICE": "cuda"}, env_path=env
    )

    lines = env.read_text(encoding="utf-8").splitlines()
    assert "# user config" in lines  # comment preserved
    assert "SEFM_LLM_N_THREADS=6" in lines  # untouched key preserved
    assert "SEFM_ACCELERATION=gpu" in lines  # existing key updated in place
    assert lines.count("SEFM_ACCELERATION=gpu") == 1  # not duplicated
    assert "SEFM_EMBEDDING_DEVICE=cuda" in lines  # new key appended


def test_update_env_file_creates_when_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    update_env_file({"SEFM_ACCELERATION": "cpu"}, env_path=env)
    assert env.exists()
    assert env.read_text(encoding="utf-8").strip() == "SEFM_ACCELERATION=cpu"
