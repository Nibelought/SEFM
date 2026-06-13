from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEFM_",
        # Absolute so reads and writes target the same file regardless of cwd.
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "models")

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # LLM
    llm_model_path: Path | None = None
    llm_n_ctx: int = 8192
    llm_n_threads: int = 0  # 0 = auto (physical core count from hardware detection)
    llm_temperature: float = 0.1
    # Output budget. Table answers with per-line citations need headroom; keep
    # well under llm_n_ctx minus the prompt.
    llm_max_tokens: int = 1536

    # Hardware acceleration
    hw_detect: bool = True  # probe CPU/GPU/NPU at startup to tune the plan
    acceleration: str = "auto"  # auto | cpu | gpu
    llm_n_gpu_layers: int = -1  # layers offloaded when GPU is active; -1 = all
    embedding_device: str = "auto"  # auto | cpu | cuda | xpu

    # Retrieval
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5
    rrf_k: int = 60

    # Chunking
    chunk_max_chars: int = 2000
    chunk_overlap_chars: int = 256

    # Safety
    offline_guard: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Path | None = None

    @property
    def resolved_log_file(self) -> Path:
        return self.log_file or (self.data_dir / "logs" / "sefm.log")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.data_dir / "bm25"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.chroma_dir, self.bm25_dir):
            p.mkdir(parents=True, exist_ok=True)


def update_env_file(updates: dict[str, str], env_path: Path | None = None) -> Path:
    """Upsert KEY=value lines into .env, preserving other lines. Keys must
    include the SEFM_ prefix. Returns the path."""
    env_path = env_path or (PROJECT_ROOT / ".env")
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    written: set[str] = set()
    out: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                written.add(key)
                continue
        out.append(line)
    for key, value in updates.items():
        if key not in written:
            out.append(f"{key}={value}")

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return env_path
