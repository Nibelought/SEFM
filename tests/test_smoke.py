"""Imports-only smoke test: catches a broken module graph without needing
any ML dependency installed."""


def test_imports() -> None:
    import app
    import app.cli
    import app.config
    import app.embedding.base
    import app.embedding.sbert
    import app.generation.base
    import app.generation.citation_guard
    import app.generation.llama_cpp
    import app.generation.prompts
    import app.hardware
    import app.indexing.keyword_store
    import app.indexing.vector_store
    import app.ingestion.chunking
    import app.ingestion.parsers.base
    import app.ingestion.parsers.pdf_parser
    import app.ingestion.pipeline
    import app.retrieval.hybrid
    import app.retrieval.types
    import app.service

    assert app.__version__


def test_settings_load() -> None:
    from app.config import Settings

    s = Settings()
    assert s.embedding_model
    assert s.top_k_final > 0
    assert s.rrf_k > 0
