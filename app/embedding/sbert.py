from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from app.embedding.base import EmbeddingModel
from app.errors import ModelLoadError
from app.logging_config import log_duration

log = logging.getLogger(__name__)


class SbertEmbedder(EmbeddingModel):
    """sentence-transformers backend. Loaded lazily on first use so the module
    imports without the dependency."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            # Lazy import: keeps the module importable without the dependency.
            try:
                from sentence_transformers import SentenceTransformer

                log.info("loading embedding model '%s' on %s", self.model_name, self.device)
                with log_duration(log, f"embedding model load '{self.model_name}'"):
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dim = int(self._model.get_sentence_embedding_dimension())
                log.info("embedding model loaded (dim=%d)", self._dim)
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load embedding model '{self.model_name}': {exc}",
                    hint=(
                        "Check the model name and that it was downloaded into the local HF cache "
                        "(first run needs network)."
                    ),
                ) from exc
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        vecs = model.encode(
            list(texts),
            normalize_embeddings=True,  # L2-normalized: cosine == dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result: list[list[float]] = vecs.tolist()
        return result
