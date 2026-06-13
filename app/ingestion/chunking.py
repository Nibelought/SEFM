from __future__ import annotations

import logging
from dataclasses import dataclass

from app.retrieval.types import Chunk

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    max_chars: int = 2000
    overlap_chars: int = 256
    keep_tables_atomic: bool = True


class Chunker:
    """Splits page-level pre-chunks into index-ready chunks.

    - Table chunks are never split when keep_tables_atomic is True, so their
      rows retrieve as a unit.
    - Text chunks split on paragraph boundaries, with char-count sizing and
      overlap carried between pieces.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def split(self, pre_chunk: Chunk) -> list[Chunk]:
        if pre_chunk.kind == "table" and self.config.keep_tables_atomic:
            return [pre_chunk]
        if len(pre_chunk.text) <= self.config.max_chars:
            return [pre_chunk]
        return self._split_text(pre_chunk)

    def _split_text(self, pre_chunk: Chunk) -> list[Chunk]:
        paragraphs = [p for p in pre_chunk.text.split("\n\n") if p.strip()]
        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for para in paragraphs:
            if buf_len + len(para) + 2 > self.config.max_chars and buf:
                pieces.append("\n\n".join(buf))
                # Carry the tail of the previous piece as overlap.
                tail = pieces[-1][-self.config.overlap_chars :] if self.config.overlap_chars else ""
                buf = [tail, para] if tail else [para]
                buf_len = sum(len(s) for s in buf)
            else:
                buf.append(para)
                buf_len += len(para) + 2
        if buf:
            pieces.append("\n\n".join(buf))

        result = [
            Chunk(
                chunk_id=f"{pre_chunk.chunk_id}#{i}",
                text=piece,
                source=pre_chunk.source,
                page=pre_chunk.page,
                kind=pre_chunk.kind,
            )
            for i, piece in enumerate(pieces)
        ]
        if len(result) > 1:
            log.debug("split chunk %s into %d pieces", pre_chunk.chunk_id, len(result))
        return result
