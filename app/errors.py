"""SEFM exception hierarchy.

Deliberate errors derive from SefmError and carry a message plus an optional
hint. The CLI and UI catch SefmError for a clean message; other exceptions
propagate as bugs.
"""
from __future__ import annotations


class SefmError(Exception):
    """Base for SEFM's deliberate errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message} ({self.hint})" if self.hint else self.message


class ConfigurationError(SefmError):
    """Invalid or missing configuration / environment."""


class LlmNotConfiguredError(ConfigurationError):
    """An LLM-requiring operation was attempted with no model configured."""


class ModelLoadError(SefmError):
    """A model file exists but could not be loaded (GGUF or sentence-transformers)."""


class IngestionError(SefmError):
    """Umbrella for ingestion-pipeline failures."""


class DocumentParseError(IngestionError):
    """A document could not be parsed. `source` is the filename, so the
    pipeline can skip-and-report it."""

    def __init__(self, message: str, *, source: str | None = None, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.source = source


class StoreError(SefmError):
    """Vector store or keyword store failed (read, write, or query)."""


class RetrievalError(SefmError):
    """Hybrid retrieval failed."""
