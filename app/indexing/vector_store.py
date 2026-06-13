from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.errors import StoreError
from app.logging_config import log_duration
from app.retrieval.types import Chunk, Hit

log = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None: ...

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[Hit]: ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None: ...


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection. Lazy import keeps tests light."""

    def __init__(self, persist_dir: Path, collection: str = "sefm") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection
        self._client: Any = None
        self._collection: Any = None

    def _ensure(self) -> Any:
        if self._collection is None:
            # Lazy import: keeps tests free of the chromadb dependency.
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self.persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(self.collection_name)
            except Exception as exc:
                raise StoreError(
                    f"Could not open the vector store at {self.persist_dir}: {exc}",
                    hint=(
                        "Delete the data\\chroma folder to rebuild the index if it is corrupt."
                    ),
                ) from exc
        return self._collection

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        log.debug("chroma upsert: %d chunks", len(chunks))
        col = self._ensure()
        with log_duration(log, f"chroma upsert ({len(chunks)} chunks)", level=logging.DEBUG):
            col.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=list(embeddings),
                documents=[c.text for c in chunks],
                metadatas=[
                    {"source": c.source, "page": c.page, "kind": c.kind} for c in chunks
                ],
            )

    def query(self, embedding: list[float], top_k: int) -> list[Hit]:
        log.debug("chroma query: top_k=%d", top_k)
        col = self._ensure()
        res = col.query(query_embeddings=[embedding], n_results=top_k)
        # Guard empty result sets: indexing [0] on an empty list raises.
        id_rows = res.get("ids") or []
        if not id_rows or not id_rows[0]:
            log.debug("chroma query returned 0 hits")
            return []
        ids = id_rows[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[Hit] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            chunk = Chunk(
                chunk_id=cid,
                text=doc,
                source=str(meta["source"]),
                page=int(meta["page"]),
                kind=str(meta.get("kind", "text")),  # type: ignore[arg-type]
            )
            # L2 distance on normalized vectors -> rough similarity for display
            # (RRF only uses rank).
            hits.append(Hit(chunk=chunk, score=1.0 - float(dist)))
        log.debug("chroma query returned %d hits", len(hits))
        return hits

    def delete_by_source(self, source: str) -> None:
        log.debug("chroma delete_by_source: %s", source)
        col = self._ensure()
        col.delete(where={"source": source})
