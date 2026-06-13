from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from app.embedding.base import EmbeddingModel
from app.errors import DocumentParseError
from app.indexing.keyword_store import KeywordStore
from app.indexing.vector_store import VectorStore
from app.ingestion.chunking import Chunker
from app.ingestion.parsers.base import DocumentParser
from app.logging_config import log_duration
from app.retrieval.types import Chunk

log = logging.getLogger(__name__)


class IngestionPipeline:
    """parse -> chunk -> embed -> index, for one path or a folder."""

    def __init__(
        self,
        parsers: list[DocumentParser],
        chunker: Chunker,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
    ) -> None:
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.skipped: list[tuple[str, str]] = []

    def run(self, path: Path) -> int:
        self.skipped = []
        files = self._iter_files(path)
        total = 0
        with log_duration(log, f"pipeline.run({path.name})"):
            for f in files:
                parser = self._pick_parser(f)
                if parser is None:
                    log.debug("no parser for %s, skipping", f.name)
                    self.skipped.append((f.name, "unsupported file type"))
                    continue
                log.info("ingesting %s", f)
                try:
                    with log_duration(log, f"ingest {f.name}", level=logging.DEBUG):
                        chunks = self._chunks_from(parser, f)
                        if not chunks:
                            log.warning("no chunks produced from %s", f.name)
                            self.skipped.append((f.name, "no extractable content"))
                            continue
                        log.debug("%s: %d chunks", f.name, len(chunks))
                        embeddings = self.embedder.encode([c.text for c in chunks])
                        self.vector_store.upsert(chunks, embeddings)
                        self.keyword_store.upsert(chunks)
                        total += len(chunks)
                except DocumentParseError as exc:
                    log.warning(
                        "skipping %s: %s",
                        f.name,
                        exc,
                    )
                    self.skipped.append((f.name, exc.message))
                except Exception as exc:
                    log.exception("unexpected error ingesting %s", f.name)
                    short = type(exc).__name__
                    self.skipped.append((f.name, short))
        log.debug("pipeline.run complete: %d total chunks", total)
        return total

    def _chunks_from(self, parser: DocumentParser, path: Path) -> list[Chunk]:
        out: list[Chunk] = []
        for pre in parser.parse(path):
            out.extend(self.chunker.split(pre))
        return out

    def _pick_parser(self, path: Path) -> DocumentParser | None:
        for p in self.parsers:
            if p.supports(path):
                return p
        return None

    @staticmethod
    def _iter_files(path: Path) -> Iterable[Path]:
        if path.is_file():
            yield path
            return
        for f in path.rglob("*"):
            if f.is_file():
                yield f
