from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.errors import StoreError
from app.logging_config import log_duration
from app.retrieval.types import Chunk, Hit

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[._-][A-Za-z0-9_]+)*|0x[0-9A-Fa-f]+")


class KeywordStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def query(self, text: str, top_k: int) -> list[Hit]: ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None: ...

    @abstractmethod
    def list_sources(self) -> dict[str, int]:
        """source filename -> chunk count."""


def tokenize(text: str) -> list[str]:
    """Lowercased tokens, preserving hex literals (0x21), part numbers
    (6ES7-XXX), and register addresses (0x2001)."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


class Bm25sStore(KeywordStore):
    """BM25 over chunk texts. Chunks are persisted as JSONL (the source of
    truth) and the index is rebuilt on load, avoiding coupling to a bm25s
    on-disk format."""

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_path = persist_dir / "chunks.jsonl"
        self._chunks: dict[str, Chunk] = {}
        self._retriever: Any = None
        self._ordered_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self._chunks_path.exists():
            log.debug("bm25 store: no existing chunks file at %s", self._chunks_path)
            return
        try:
            with self._chunks_path.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        chunk = Chunk(**row)
                        self._chunks[chunk.chunk_id] = chunk
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        log.warning(
                            "bm25 store: skipping malformed line %d in %s (%s)",
                            lineno,
                            self._chunks_path,
                            exc,
                        )
        except OSError as exc:
            log.error(
                "bm25 store: could not read %s (%s); starting with empty index",
                self._chunks_path,
                exc,
            )
            return
        log.debug("bm25 store: loaded %d chunks from disk", len(self._chunks))
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._chunks:
            self._retriever = None
            self._ordered_ids = []
            return
        import bm25s  # type: ignore[import-untyped]

        with log_duration(log, f"bm25 index rebuild ({len(self._chunks)} docs)", level=logging.DEBUG):
            self._ordered_ids = list(self._chunks.keys())
            corpus_tokens = [tokenize(self._chunks[cid].text) for cid in self._ordered_ids]
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens, show_progress=False)
            self._retriever = retriever
        log.debug("bm25 index rebuilt: %d documents", len(self._ordered_ids))

    def _persist(self) -> None:
        # A failed write IS surfaced: the on-disk store would be out of sync
        # and a restart would lose these chunks.
        try:
            tmp = self._chunks_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for chunk in self._chunks.values():
                    f.write(
                        json.dumps(
                            {
                                "chunk_id": chunk.chunk_id,
                                "text": chunk.text,
                                "source": chunk.source,
                                "page": chunk.page,
                                "kind": chunk.kind,
                            }
                        )
                        + "\n"
                    )
            tmp.replace(self._chunks_path)
        except OSError as exc:
            log.error("bm25 store: could not write %s (%s)", self._chunks_path, exc)
            raise StoreError(
                f"Could not persist keyword index to {self._chunks_path}: {exc}",
                hint="Check disk space and permissions on the data directory.",
            ) from exc

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c
        log.debug("bm25 upsert: +%d chunks, total=%d", len(chunks), len(self._chunks))
        self._persist()
        self._rebuild()

    def query(self, text: str, top_k: int) -> list[Hit]:
        if self._retriever is None or not self._ordered_ids:
            return []
        q_tokens = tokenize(text)
        if not q_tokens:
            return []
        k = min(top_k, len(self._ordered_ids))
        results, scores = self._retriever.retrieve([q_tokens], k=k, show_progress=False)
        hits: list[Hit] = []
        for idx, score in zip(results[0].tolist(), scores[0].tolist(), strict=True):
            cid = self._ordered_ids[int(idx)]
            hits.append(Hit(chunk=self._chunks[cid], score=float(score)))
        return hits

    def delete_by_source(self, source: str) -> None:
        before = len(self._chunks)
        self._chunks = {cid: c for cid, c in self._chunks.items() if c.source != source}
        removed = before - len(self._chunks)
        if removed:
            log.debug("bm25 delete_by_source '%s': removed %d chunks", source, removed)
            self._persist()
            self._rebuild()

    def list_sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._chunks.values():
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts
