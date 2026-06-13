# SEFM – Offline RAG Assistant for ICS/SCADA Documentation

SEFM is a desktop RAG (Retrieval-Augmented Generation) application for ICS/SCADA technical documentation. Engineers ingest PDF manuals from vendors (Siemens, Schneider, Phoenix Contact, Eaton, CATL) and ask natural-language questions. A quantized LLM running on CPU answers strictly from the indexed corpus with mandatory page-level citations.

Everything runs **locally and offline** — no network at runtime — because field sites are air-gapped.

## Features

- **Local-first.** All inference on CPU via quantized models; zero external dependencies.
- **Hybrid retrieval.** Dense + sparse search (RRF fusion) for robust multi-modal recall.
- **Mandatory citations.** Every answer cites its sources with page numbers, validated against the indexed chunks.
- **Bulk ingest.** Drag folders of PDFs into the library; corrupt files skip gracefully with warnings.
- **Desktop UI.** PySide6 GUI with tabs for Library, Search, and Ask; threaded workers prevent UI freeze.
- **CLI fallback.** Full programmatic access via command-line tools for scripting and testing.

## Requirements

- **Python 3.13+** (tested on 3.14)
- **Windows** with PowerShell
- **~4–8 GB RAM** (depends on model size and corpus)
- Optional: **GPU/CUDA** for faster inference (auto-detected; CPU fallback always available)

## Quick Start

### 1. Install

```powershell
# Clone and enter the repo
git clone https://github.com/your-org/sefm.git
cd sefm

# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (including llama-cpp-python CPU wheels)
S:\SEFM\.venv\Scripts\python.exe -m pip install -e ".[dev]"
S:\SEFM\.venv\Scripts\python.exe -m pip install llama-cpp-python `
    --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. Configure

Create a `.env` file in the project root to override defaults:

```
SEFM_LLM_MODEL_PATH=models/my-model-q4.gguf
SEFM_LOG_LEVEL=INFO
```

See `app/config.py` for all available settings.

### 3. Ingest PDFs

```powershell
# Bulk ingest a folder
S:\SEFM\.venv\Scripts\python.exe -m app ingest pdfs C:\path\to\manuals

# Check system info and model paths
S:\SEFM\.venv\Scripts\python.exe -m app info
S:\SEFM\.venv\Scripts\python.exe -m app hardware
```

### 4. Search and Ask

```powershell
# Hybrid search (no LLM, just retrieval)
S:\SEFM\.venv\Scripts\python.exe -m app search "fault 0x21" -n 5

# Full RAG with citation validation
S:\SEFM\.venv\Scripts\python.exe -m app ask "What causes error code 0x21?"

# GUI (requires SEFM_LLM_MODEL_PATH set)
S:\SEFM\.venv\Scripts\python.exe -m app gui
```

## Project Structure

```
sefm/
├── app/
│   ├── service.py              # Composition root (AppService)
│   ├── factory.py              # Dependency injection
│   ├── config.py               # Settings schema + .env parsing
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
│   │   ├── keyword_store.py    # Bm25sStore + tokenization
│   │   └── source_registry.py  # PDF path registry (JSON)
│   │
│   ├── retrieval/
│   │   ├── hybrid.py           # RRF fusion logic
│   │   └── ranking.py          # Reranking utilities
│   │
│   ├── generation/
│   │   ├── llm_backend.py      # LLMBackend ABC + llama-cpp impl
│   │   ├── prompts.py          # Prompt templates
│   │   └── citation_guard.py   # Citation validation + refusal
│   │
│   └── ui/
│       ├── main_window.py      # MainWindow + tab layout
│       ├── library_view.py     # Library management UI
│       ├── search_view.py      # Search + retrieval preview
│       ├── ask_view.py         # Ask with streaming + citations
│       ├── workers.py          # QThreadPool task runners
│       ├── dialogs.py          # Settings, ingestion, errors
│       └── run.py              # GUI entry point
│
├── tests/
│   ├── test_*.py               # Unit + integration tests
│   └── fixtures/               # Sample PDFs, mock data
│
├── scripts/
│   ├── debug_ask.py            # Debug tool for Q&A pipeline
│   ├── make_sample_pdf.py      # Regenerate test fixtures
│   ├── ui_smoke.py             # GUI screenshot tests
│   └── ui_smoke_ask.py         # Full Ask path smoke test
│
├── docs/
│   ├── ANALYSIS.md             # Design rationale + risk register
│   ├── FOUNDATION_PLAN.md      # Phase 0 foundations
│   ├── PROJECT_PLAN.md         # Multi-phase roadmap
│   └── ...
│
├── .venv/                      # Python virtual environment
├── models/                     # GGUF quantized models (gitignored)
├── pdfs/                       # Ingested PDF corpus (gitignored)
├── data/                       # Chroma + BM25 indices + registry (gitignored)
├── .env                        # Local config overrides (gitignored)
├── CLAUDE.md                   # Detailed architecture guide for Claude Code
├── pyproject.toml              # Package metadata + dev deps
└── README.md                   # This file
```

## Testing

```powershell
# Run all tests
S:\SEFM\.venv\Scripts\python.exe -m pytest tests -q

# Run a specific test class
S:\SEFM\.venv\Scripts\python.exe -m pytest tests/test_citation_guard.py::TestValidateAnswer -v

# Filter by keyword
S:\SEFM\.venv\Scripts\python.exe -m pytest tests -k "hybrid"

# Lint and type-check
S:\SEFM\.venv\Scripts\python.exe -m ruff check app tests
S:\SEFM\.venv\Scripts\python.exe -m mypy app
```

## Debug Tools

- **`scripts/debug_ask.py`** — Shows retrieved chunks, raw LLM output, and citation-guard verdict for a question. Best tool for "why did the model say X?"
- **`app search`** — Hybrid retrieval without generation; confirms chunks surface before debugging generation.
- **`app info`** — Prints config, model paths, and acceleration plan.
- **`app hardware`** — Detects CPU/GPU/NPU and recommends install steps if acceleration is available but missing.

## Architecture Highlights

- **Composition root (`AppService`):** Single entry point for ingest, search, ask; used by CLI, UI, and tests.
- **Abstraction boundaries (ABCs):** `DocumentParser`, `Chunker`, `EmbeddingModel`, `VectorStore`, `KeywordStore`, `LLMBackend` — swap implementations without touching calling code.
- **Citation contract:** Prompts request `[N]` citations, expanded to `[source: file, p. N]` post-generation and validated against retrieved chunks; ungrounded answers are rejected.
- **Lazy ML imports:** Heavy dependencies (sentence-transformers, chromadb, llama-cpp-python) imported inside `_ensure()` / `_load()` methods so tests and CLI can run without them.
- **Offline-only:** No network at runtime; setup pulls embedding models from Hugging Face on first run, but that's a setup step.
- **Graceful degradation:** Registry and skipped files degrade (warnings, not crashes); corrupt PDFs are recorded and skip-continued.

## For More Information

- **`CLAUDE.md`** — Detailed architecture, design contracts, layer notes, and instructions for Claude Code.
- **`docs/ANALYSIS.md`** — Design rationale, risk register, and motivation.
- **`docs/FOUNDATION_PLAN.md`** — Phase 0 (MVP) architecture and deliverables.
- **`docs/PROJECT_PLAN.md`** — Multi-phase roadmap and feature priorities.

## License

[Add your license here]

## Contact

For questions, bugs, or contributions, see the issue tracker or contact the maintainers.
