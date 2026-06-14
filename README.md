# SEFM

Offline RAG assistant for ICS/SCADA technical documentation. Ingest PDFs from vendors (Siemens, Schneider, Phoenix Contact, Eaton, CATL), then ask natural-language questions. A quantized LLM answers from the indexed corpus with page-level citations — everything runs locally on CPU, no network at runtime.

## Features

- CPU inference only; no external dependencies
- Hybrid retrieval (dense + sparse search via RRF)
- Mandatory citations validated against chunks
- Bulk PDF ingest with graceful error handling
- PySide6 GUI + CLI

## Install

Requires Python 3.13+ on Windows.

```powershell
git clone https://github.com/your-org/sefm.git
cd sefm

python -m venv .venv
.\.venv\Scripts\Activate.ps1

S:\SEFM\.venv\Scripts\python.exe -m pip install -e ".[dev]"
S:\SEFM\.venv\Scripts\python.exe -m pip install llama-cpp-python `
    --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Set `SEFM_LLM_MODEL_PATH` in `.env` to point to a GGUF model, then:

```powershell
# Ingest PDFs
S:\SEFM\.venv\Scripts\python.exe -m app ingest pdfs C:\path\to\manuals

# Search (no LLM)
S:\SEFM\.venv\Scripts\python.exe -m app search "fault 0x21" -n 5

# Full RAG
S:\SEFM\.venv\Scripts\python.exe -m app ask "What causes error 0x21?"

# GUI
S:\SEFM\.venv\Scripts\python.exe -m app gui
```

See `app/config.py` for all settings and `scripts/debug_ask.py` for inspecting the Q&A pipeline.

## Project Structure

```
sefm/
├── app/
│   ├── service.py              # Composition root (AppService)
│   ├── factory.py              # Dependency injection
│   ├── config.py               # Settings schema + defaults
│   ├── logging_config.py        # Logging setup
│   ├── errors.py               # Error hierarchy
│   ├── hardware.py             # CPU/GPU/NPU detection
│   ├── cli.py                  # CLI entry points
│   │
│   ├── ingestion/
│   │   ├── parsers/            # DocumentParser impls (PDF, DOCX, ...)
│   │   ├── chunking.py         # Chunk splitting logic
│   │   └── pipeline.py         # Bulk ingest orchestration
│   │
│   ├── indexing/
│   │   ├── embedding.py        # EmbeddingModel ABC + SBERT impl
│   │   ├── vector_store.py     # VectorStore ABC + Chroma impl
│   │   ├── keyword_store.py    # BM25 keyword index
│   │   └── source_registry.py  # PDF path registry
│   │
│   ├── retrieval/
│   │   ├── hybrid.py           # RRF fusion (dense + sparse)
│   │   └── ranking.py          # Reranking utilities
│   │
│   ├── generation/
│   │   ├── llm_backend.py      # LLMBackend ABC + llama-cpp impl
│   │   ├── prompts.py          # Prompt templates
│   │   └── citation_guard.py   # Citation validation
│   │
│   └── ui/
│       ├── main_window.py      # MainWindow + tab layout
│       ├── library_view.py     # Library management
│       ├── search_view.py      # Search + retrieval preview
│       ├── ask_view.py         # Ask with streaming
│       ├── workers.py          # QThreadPool task runners
│       ├── dialogs.py          # Settings, ingestion, errors
│       └── run.py              # GUI entry point
│
├── tests/
│   ├── test_*.py               # Unit + integration tests
│   └── fixtures/               # Sample test data
│
├── scripts/
│   ├── debug_ask.py            # Debug Q&A pipeline
│   ├── make_sample_pdf.py      # Regenerate test fixtures
│   ├── ui_smoke.py             # GUI screenshot tests
│   └── ui_smoke_ask.py         # Full Ask path test
│
├── docs/
│   ├── ANALYSIS.md             # Design rationale
│   ├── FOUNDATION_PLAN.md      # Phase 0 architecture
│   ├── PROJECT_PLAN.md         # Roadmap
│   └── ...
│
├── CLAUDE.md                   # Architecture guide for Claude Code
├── pyproject.toml              # Package metadata + dev deps
└── README.md                   # This file
```

## Test & Debug

```powershell
S:\SEFM\.venv\Scripts\python.exe -m pytest tests -q
S:\SEFM\.venv\Scripts\python.exe -m ruff check app tests
S:\SEFM\.venv\Scripts\python.exe -m mypy app
```

Use `scripts/debug_ask.py "<question>"` to inspect retrieved chunks, raw LLM output, and citation validation. Run `app search` to test retrieval alone before debugging generation.

## Architecture

Single composition root (`AppService`) with abstract boundaries (`DocumentParser`, `EmbeddingModel`, `VectorStore`, `LLMBackend`) so implementations swap cleanly. Citation contract: prompt requests `[N]`, expanded post-generation to `[source: file, p. N]` and validated against retrieved chunks. Heavy dependencies (sentence-transformers, chromadb, llama-cpp-python) are lazily imported so tests and CLI work without them.

See `CLAUDE.md` for detailed architecture and design contracts. `docs/ANALYSIS.md`, `docs/FOUNDATION_PLAN.md`, `docs/PROJECT_PLAN.md` cover rationale, phase plan, and roadmap.
