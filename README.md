# SEFM

Offline RAG assistant for ICS/SCADA technical documentation. Drop in vendor PDFs from Siemens, Schneider, Phoenix Contact, Eaton, or CATL; ask natural-language questions; get grounded answers with mandatory page-level citations — all on CPU, no network required at runtime.

## Features

- **Fully offline at runtime** — no telemetry, no auto-updates, designed for air-gapped field sites
- **Hybrid retrieval** — dense (SBERT) + sparse (BM25) search fused via reciprocal rank fusion
- **Mandatory citations** — every answer is validated against the retrieved chunks; uncited or hallucinated claims are rejected outright
- **Tables stay atomic** — fault tables and register maps are indexed as single chunks, so hex-code lookups return the full table
- **Bulk ingest with graceful errors** — a corrupt or encrypted PDF is skipped with a warning; the rest of the folder still ingests
- **PySide6 GUI + CLI** — click a citation in the Ask tab to jump directly to the cited page in the built-in PDF viewer

## Requirements

- Windows, Python 3.13+
- A GGUF model file (any llama.cpp–compatible model: Phi-3.5, Gemma 3n, Qwen, Mistral, …)

## Install

```powershell
git clone <repo-url>
cd sefm

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[all]"

# llama-cpp-python has no standard PyPI wheel; use the prebuilt CPU index
pip install llama-cpp-python --prefer-binary `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Copy `.env.example` to `.env` and set `SEFM_LLM_MODEL_PATH` to the path of your GGUF file.

## Quick start

```powershell
# Check your configuration
sefm info

# Ingest a folder of PDFs
sefm ingest C:\path\to\manuals

# Search the index without the LLM
sefm search "fault 0x21" -n 5

# Ask a question (requires SEFM_LLM_MODEL_PATH)
sefm ask "What causes error 0x21?"

# Open the GUI
sefm gui
```

`sefm` is available after `pip install -e ".[all]"`. Alternatively, use `python -m app <command>` without activating the virtualenv.

## Configuration

Copy `.env.example` to `.env`. All settings are optional — defaults are in `app/config.py`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `SEFM_LLM_MODEL_PATH` | _(required for ask/gui)_ | Path to your GGUF model file |
| `SEFM_DATA_DIR` | `./data` | Index, registry, and log storage |
| `SEFM_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `SEFM_LLM_N_CTX` | `4096` | Context window size |
| `SEFM_LLM_N_THREADS` | `4` | CPU threads for inference |

Run `sefm hardware` to see the detected hardware and the chosen acceleration plan, including the exact `pip` commands to unlock GPU offload if available.

## Project structure

```
sefm/
├── app/
│   ├── service.py              # Composition root (AppService)
│   ├── factory.py              # Dependency injection / wiring
│   ├── config.py               # Settings schema (pydantic-settings, SEFM_ prefix)
│   ├── errors.py               # Error hierarchy (SefmError + subclasses)
│   ├── hardware.py             # CPU/GPU/NPU detection and acceleration planning
│   ├── registry.py             # Source path registry (PDF path → disk path)
│   ├── logging_config.py       # Logging setup (console + rotating file)
│   ├── cli.py                  # CLI entry points (typer)
│   │
│   ├── ingestion/
│   │   ├── parsers/
│   │   │   ├── base.py         # DocumentParser ABC
│   │   │   └── pdf_parser.py   # PyMuPDF + pdfplumber parser
│   │   ├── chunking.py         # Chunk splitting (tables kept atomic)
│   │   └── pipeline.py         # Bulk ingest orchestration
│   │
│   ├── embedding/
│   │   ├── base.py             # EmbeddingModel ABC
│   │   └── sbert.py            # SBERT implementation (lazy import)
│   │
│   ├── indexing/
│   │   ├── vector_store.py     # VectorStore ABC + Chroma implementation
│   │   └── keyword_store.py    # BM25 keyword index (bm25s)
│   │
│   ├── retrieval/
│   │   ├── hybrid.py           # RRF fusion (dense + sparse)
│   │   └── types.py            # Chunk, Hit, Citation dataclasses
│   │
│   ├── generation/
│   │   ├── base.py             # LLMBackend ABC
│   │   ├── llama_cpp.py        # llama-cpp-python backend (lazy import)
│   │   ├── prompts.py          # Prompt templates
│   │   └── citation_guard.py   # Citation expansion and answer validation
│   │
│   └── ui/
│       ├── main_window.py      # MainWindow + tab layout
│       ├── library_view.py     # Library tab (ingest, list, remove documents)
│       ├── search_view.py      # Search tab (retrieval preview)
│       ├── ask_view.py         # Ask tab (streaming answers + citation links)
│       ├── pdf_viewer.py       # PDF viewer dialog (opens at cited page)
│       ├── settings_dialog.py  # Settings + acceleration configuration
│       ├── workers.py          # QThreadPool task runners
│       └── run.py              # GUI entry point
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/sample_vfd.pdf # Minimal test PDF
│   ├── test_chunking.py
│   ├── test_citation_guard.py
│   ├── test_config.py
│   ├── test_hardware.py
│   ├── test_hybrid.py
│   ├── test_prompts.py
│   └── test_smoke.py
│
├── scripts/
│   ├── debug_ask.py            # Inspect chunks, raw LLM output, guard verdict
│   ├── make_sample_pdf.py      # Regenerate tests/fixtures/sample_vfd.pdf
│   ├── ui_smoke.py             # Screenshot Library + Search tabs
│   └── ui_smoke_ask.py         # Drive the full Ask path, screenshot
│
├── .env.example                # Configuration template
├── pyproject.toml              # Package metadata and dependencies
└── README.md                   # This file
```

Heavy artifacts live outside the source tree and are gitignored: `models/` (GGUF files), `pdfs/` (your corpus), `data/` (Chroma index, BM25 index, registry, logs).

## Development

```powershell
# Install dev tools alongside the full package
pip install -e ".[all,dev]"

# Run the test suite
python -m pytest tests -q
python -m pytest tests/test_citation_guard.py::TestValidateAnswer -v   # single class
python -m pytest tests -k "hybrid"                                      # by keyword

# Lint and typecheck
python -m ruff check app tests
python -m mypy app
```

### Debugging

`scripts/debug_ask.py` is the single best tool for tracing a bad answer:

```powershell
python scripts/debug_ask.py "What causes error 0x21?"
```

It prints the retrieved chunks, the tail of the prompt, the raw LLM output, and the citation-guard verdict. If the guard is rejecting a valid-looking answer, inspect the raw output — the most common cause is the model writing `[Source 1]` instead of `[1]`, which the expander can't match.

If retrieval itself looks wrong, test it in isolation before touching generation:

```powershell
sefm search "fault 0x21" -n 5
```

## Architecture

Single composition root (`AppService`) with abstract boundaries at every swap point — `DocumentParser`, `EmbeddingModel`, `VectorStore`, `LLMBackend` — so implementations change without touching call sites. `app/factory.py` wires everything from `Settings`; the CLI and UI only ever talk to `AppService`.

**Citation contract:** the prompt asks the model for `[N]` (1-indexed into the context blocks). Post-generation, `expand_numeric_citations` rewrites `[N]` → `[source: <filename>, p. <page>]`. `validate_answer` then checks every cited `(source, page)` was actually in the retrieved set and that the answer has at least one citation; otherwise it returns the refusal string `"Not found in the indexed documents."`.

**Lazy imports:** `SbertEmbedder`, `ChromaVectorStore`, `Bm25sStore`, and `LlamaCppBackend` load their heavy dependencies inside `_ensure()` / `_load()`, not at module level. `pytest` and `sefm info` run without sentence-transformers, chromadb, or llama-cpp-python installed.

See `CLAUDE.md` for detailed design contracts and layer notes (gitignored; present in working copies only).
