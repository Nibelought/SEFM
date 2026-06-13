from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str: ...

    @abstractmethod
    def stream(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.1
    ) -> Iterator[str]: ...
