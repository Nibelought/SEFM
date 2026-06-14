from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.embedding.base import EmbeddingModel
from app.errors import LlmNotConfiguredError, RetrievalError, SefmError
from app.generation.base import LLMBackend
from app.generation.citation_guard import parse_citations, validate_answer
from app.generation.prompts import build_prompt
from app.indexing.keyword_store import KeywordStore
from app.indexing.vector_store import VectorStore
from app.ingestion.pipeline import IngestionPipeline, ProgressCallback
from app.logging_config import log_duration
from app.registry import SourceRegistry
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.types import Chunk, Citation, Hit

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    used_chunks: list[Chunk]


class AppService:
    """Composition root. The UI and CLI talk to this; everything else is plumbing."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        ingestion: IngestionPipeline,
        registry: SourceRegistry,
        llm: LLMBackend | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.ingestion = ingestion
        self.registry = registry
        self.llm = llm

    def ingest_path(
        self, path: Path, progress: ProgressCallback | None = None
    ) -> int:
        path = path.resolve()
        log.info("ingest_path: %s", path)
        n = self.ingestion.run(path, progress=progress)
        if path.is_file():
            self.registry.remember(path.name, path)
        else:
            for f in path.rglob("*.pdf"):
                if f.is_file():
                    self.registry.remember(f.name, f)
        skipped = getattr(self.ingestion, "skipped", [])
        if skipped:
            log.warning(
                "ingest skipped %d file(s): %s",
                len(skipped),
                ", ".join(f"{name} ({reason})" for name, reason in skipped),
            )
        log.info("ingest_path complete: %d chunks indexed", n)
        return n

    def list_documents(self) -> dict[str, int]:
        return self.keyword_store.list_sources()

    def remove_document(self, source: str) -> None:
        log.info("remove_document: %s", source)
        self.vector_store.delete_by_source(source)
        self.keyword_store.delete_by_source(source)
        self.registry.forget(source)

    def resolve_source_path(self, source: str) -> Path | None:
        return self.registry.get(source)

    def search(self, question: str, top_n: int | None = None) -> list[Hit]:
        top_n = top_n or self.settings.top_k_final
        log.debug("search: top_n=%d  question=%r", top_n, question[:120])
        with log_duration(log, f"search {question[:60]!r}"):
            try:
                q_emb = self.embedder.encode([question])[0]
                dense = self.vector_store.query(q_emb, self.settings.top_k_dense)
                sparse = self.keyword_store.query(question, self.settings.top_k_bm25)
                hits = reciprocal_rank_fusion(dense, sparse, k=self.settings.rrf_k, top_n=top_n)
            except SefmError:
                raise
            except Exception as e:
                raise RetrievalError(
                    f"Retrieval failed: {e}",
                    hint="Check that the vector store and keyword store are initialised (run `sefm ingest` first).",
                ) from e
        log.debug("search returned %d hits", len(hits))
        return hits

    def ask(self, question: str) -> AnswerResult:
        if self.llm is None:
            raise LlmNotConfiguredError(
                "No LLM backend is configured.",
                hint="Set SEFM_LLM_MODEL_PATH to a GGUF file, or use search() for retrieval-only.",
            )
        log.info("ask: %r", question[:120])
        with log_duration(log, f"ask {question[:60]!r}"):
            fused = self.search(question)
            retrieved = [hit.chunk for hit in fused]
            log.debug("ask: %d chunks retrieved", len(retrieved))
            for i, hit in enumerate(fused, 1):
                log.debug("  chunk[%d] %s p.%d  rrf=%.4f", i, hit.chunk.source, hit.chunk.page, hit.score)

            prompt = build_prompt(question, retrieved)
            log.debug(
                "ask: prompt (%d chars)\n%s\n--- end prompt ---",
                len(prompt),
                prompt,
            )

            raw = self.llm.generate(
                prompt,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
            )
            log.debug(
                "ask: raw LLM output (%d chars)\n%s\n--- end raw output ---",
                len(raw),
                raw,
            )

            validated = validate_answer(raw, retrieved)
            log.debug(
                "ask: validated output (%d chars)\n%s\n--- end validated output ---",
                len(validated),
                validated,
            )
            if validated == "Not found in the indexed documents.":
                log.warning("ask: answer refused by citation guard")
            else:
                log.info("ask: answer accepted (%d chars)", len(validated))
        return AnswerResult(
            answer=validated,
            citations=parse_citations(validated),
            used_chunks=retrieved,
        )
