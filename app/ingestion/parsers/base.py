from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from app.retrieval.types import Chunk


class DocumentParser(ABC):
    """Parses a document into per-page pre-chunks. A "page" is the smallest
    citable unit. The Chunker may split these while preserving (source, page)."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """True if this parser can handle the file."""

    @abstractmethod
    def parse(self, path: Path) -> Iterator[Chunk]:
        """Yield pre-chunks. Tables arrive as kind="table" with text serialized
        to Markdown so the chunker keeps them atomic."""
