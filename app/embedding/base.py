from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingModel(ABC):
    """A text embedder. Implementations must return L2-normalized vectors so
    cosine reduces to dot product. Batching is the impl's responsibility."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality (stable per instance)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in input order."""
