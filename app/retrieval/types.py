from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChunkKind = Literal["text", "table", "ocr"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """An indexable unit of a document. chunk_id must be stable across re-runs
    (e.g. f"{source}#{page}#{ordinal}") so re-ingest replaces, not duplicates."""

    chunk_id: str
    text: str
    source: str
    page: int
    kind: ChunkKind = "text"


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    source: str
    page: int
