from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.errors import ModelLoadError, SefmError
from app.generation.base import LLMBackend

log = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    """Quantized GGUF model via llama-cpp-python.

    Uses create_chat_completion so the GGUF's embedded chat template handles
    per-model formatting; the prompt is passed as a single user message. Loaded
    lazily on first generate/stream so the module imports without the dependency.
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 8192,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
        seed: int = 0,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        # GPU layers: 0 = CPU, -1 = all. Effective only with a GPU-enabled wheel
        # (gated by the acceleration planner).
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self._llm: Any = None

    def _ensure(self) -> Any:
        if self._llm is None:
            from llama_cpp import Llama

            if not self.model_path.exists():
                raise ModelLoadError(
                    f"GGUF model not found at {self.model_path}.",
                    hint="Set SEFM_LLM_MODEL_PATH to a valid GGUF file.",
                )
            log.info(
                "loading GGUF model: %s  (n_ctx=%d, n_threads=%d, n_gpu_layers=%d)",
                self.model_path.name,
                self.n_ctx,
                self.n_threads,
                self.n_gpu_layers,
            )
            try:
                self._llm = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    seed=self.seed,
                    verbose=False,
                )
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load GGUF model '{self.model_path.name}': {e}",
                    hint="The file may be corrupt or an unsupported quantization; re-download it.",
                ) from e
            log.info("GGUF model loaded")
        return self._llm

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        log.debug("generate: max_tokens=%d  temperature=%.2f", max_tokens, temperature)
        llm = self._ensure()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not out.get("choices"):
            raise SefmError(
                "The LLM returned an empty response.",
                hint="Try again or lower SEFM_LLM_MAX_TOKENS / check the model.",
            )
        choice = out["choices"][0]
        text = (choice["message"]["content"] or "").strip()
        usage = out.get("usage", {})
        log.debug(
            "generate: prompt_tokens=%s  completion_tokens=%s",
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )
        if choice.get("finish_reason") == "length":
            log.warning(
                "generate: output truncated at max_tokens=%d — raise SEFM_LLM_MAX_TOKENS "
                "(and SEFM_LLM_N_CTX if needed)",
                max_tokens,
            )
        return text

    def stream(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.1
    ) -> Iterator[str]:
        log.debug("stream: max_tokens=%d  temperature=%.2f", max_tokens, temperature)
        log.debug(
            "stream: prompt (%d chars)\n%s\n--- end prompt ---",
            len(prompt),
            prompt,
        )
        llm = self._ensure()
        buf: list[str] = []
        finish_reason: str | None = None
        for chunk in llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            choices = chunk.get("choices")
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}).get("content")
            if delta:
                buf.append(delta)
                yield delta
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        full = "".join(buf)
        log.debug(
            "stream: complete raw output (%d chars, finish_reason=%s)\n%s\n"
            "--- end stream output ---",
            len(full),
            finish_reason,
            full,
        )
        if finish_reason == "length":
            log.warning(
                "stream: output truncated at max_tokens=%d — raise SEFM_LLM_MAX_TOKENS "
                "(and SEFM_LLM_N_CTX if needed)",
                max_tokens,
            )
