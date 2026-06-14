from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
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

# progress(fraction in [0, 1], human-readable label)
ProgressCallback = Callable[[float, str], None]


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

    def run(self, path: Path, progress: ProgressCallback | None = None) -> int:
        self.skipped = []
        # Materialise up front so we know the total for determinate progress.
        files = list(self._iter_files(path))
        n_files = len(files)

        def report(done_files: int, phase_frac: float, label: str) -> None:
            if progress is None or n_files == 0:
                return
            # Each file owns the slice [k/N, (k+1)/N]; phase_frac walks within it.
            frac = (done_files + phase_frac) / n_files
            progress(min(frac, 1.0), label)

        total = 0
        with log_duration(log, f"pipeline.run({path.name})"):
            for k, f in enumerate(files):
                parser = self._pick_parser(f)
                if parser is None:
                    log.debug("no parser for %s, skipping", f.name)
                    self.skipped.append((f.name, "unsupported file type"))
                    report(k + 1, 0.0, f"Skipped {f.name} ({k + 1}/{n_files})")
                    continue
                log.info("ingesting %s", f)
                try:
                    with log_duration(log, f"ingest {f.name}", level=logging.DEBUG):
                        report(k, 0.0, f"Reading {f.name} ({k + 1}/{n_files})")
                        chunks = self._chunks_from(parser, f)
                        if not chunks:
                            log.warning("no chunks produced from %s", f.name)
                            self.skipped.append((f.name, "no extractable content"))
                            report(k + 1, 0.0, f"Skipped {f.name} ({k + 1}/{n_files})")
                            continue
                        log.debug("%s: %d chunks", f.name, len(chunks))
                        report(
                            k, 0.3,
                            f"Embedding {len(chunks)} chunk(s) from {f.name} "
                            f"({k + 1}/{n_files})",
                        )
                        embeddings = self.embedder.encode([c.text for c in chunks])
                        report(k, 0.9, f"Indexing {f.name} ({k + 1}/{n_files})")
                        self.vector_store.upsert(chunks, embeddings)
                        self.keyword_store.upsert(chunks)
                        total += len(chunks)
                        report(k + 1, 0.0, f"Indexed {f.name} ({k + 1}/{n_files})")
                except DocumentParseError as exc:
                    log.warning(
                        "skipping %s: %s",
                        f.name,
                        exc,
                    )
                    self.skipped.append((f.name, exc.message))
                    report(k + 1, 0.0, f"Skipped {f.name} ({k + 1}/{n_files})")
                except Exception as exc:
                    log.exception("unexpected error ingesting %s", f.name)
                    short = type(exc).__name__
                    self.skipped.append((f.name, short))
                    report(k + 1, 0.0, f"Skipped {f.name} ({k + 1}/{n_files})")
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
