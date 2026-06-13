from __future__ import annotations

import logging

from app.config import Settings
from app.embedding.sbert import SbertEmbedder
from app.generation.base import LLMBackend
from app.generation.llama_cpp import LlamaCppBackend
from app.hardware import HardwareProfile, detect_hardware, plan_acceleration
from app.indexing.keyword_store import Bm25sStore
from app.indexing.vector_store import ChromaVectorStore
from app.ingestion.chunking import Chunker, ChunkerConfig
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.pipeline import IngestionPipeline
from app.logging_config import log_duration
from app.registry import SourceRegistry
from app.service import AppService

log = logging.getLogger(__name__)


def build_app_service(settings: Settings | None = None, *, with_llm: bool = True) -> AppService:
    settings = settings or Settings()
    settings.ensure_dirs()
    log.info("building AppService (with_llm=%s)", with_llm)

    with log_duration(log, "build AppService"):
        profile = detect_hardware() if settings.hw_detect else HardwareProfile.unknown()
        plan = plan_acceleration(profile, settings)
        log.info(plan.summary())
        for reason in plan.reasons:
            log.debug("  %s", reason)

        embedder = SbertEmbedder(settings.embedding_model, device=plan.embedding_device)
        vector_store = ChromaVectorStore(settings.chroma_dir)
        keyword_store = Bm25sStore(settings.bm25_dir)

        parsers: list[DocumentParser] = [PdfParser()]
        chunker = Chunker(
            ChunkerConfig(
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            )
        )
        ingestion = IngestionPipeline(parsers, chunker, embedder, vector_store, keyword_store)

        llm: LLMBackend | None = None
        if with_llm and settings.llm_model_path and settings.llm_model_path.exists():
            log.debug("LLM backend: %s", settings.llm_model_path)
            llm = LlamaCppBackend(
                settings.llm_model_path,
                n_ctx=settings.llm_n_ctx,
                n_threads=plan.llm_n_threads,
                n_gpu_layers=plan.llm_n_gpu_layers,
            )
        elif with_llm:
            log.warning("LLM requested but SEFM_LLM_MODEL_PATH is unset or missing")

        registry = SourceRegistry(settings.registry_path)

    return AppService(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        keyword_store=keyword_store,
        ingestion=ingestion,
        registry=registry,
        llm=llm,
    )
