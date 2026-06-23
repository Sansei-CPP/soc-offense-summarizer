"""Runtime configuration.

All settings are read from environment variables (with sensible defaults) so the
service can be reconfigured without code changes. The defaults target a local
Ollama install with the models the project was developed against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo root = parent of the package directory. Used to resolve the default
# knowledge-base location so the service works regardless of the CWD.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
_DEFAULT_KB_DIR = _REPO_ROOT / "data" / "knowledge_base"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable configuration for the summarizer service."""

    # --- Ollama connection ---
    # OLLAMA_HOST is the variable the Ollama client itself understands, so we
    # reuse it here for consistency with the rest of the toolchain.
    ollama_host: str = _env_str("OLLAMA_HOST", "http://localhost:11434")

    # --- Models ---
    llm_model: str = _env_str("SOC_LLM_MODEL", "qwen2.5:14b")
    embedding_model: str = _env_str("SOC_EMBEDDING_MODEL", "nomic-embed-text")

    # --- Knowledge base ---
    knowledge_base_dir: Path = Path(_env_str("SOC_KB_DIR", str(_DEFAULT_KB_DIR)))

    # --- Chunking (measured in characters) ---
    chunk_size: int = _env_int("SOC_CHUNK_SIZE", 800)
    chunk_overlap: int = _env_int("SOC_CHUNK_OVERLAP", 120)

    # --- Retrieval ---
    top_k: int = _env_int("SOC_TOP_K", 4)

    # --- Generation ---
    # Low temperature: triage decisions should be stable and reproducible.
    temperature: float = _env_float("SOC_TEMPERATURE", 0.1)
    # Context window for the chat model. The assembled prompt is small, but we
    # give plenty of headroom so retrieved context is never silently truncated.
    num_ctx: int = _env_int("SOC_NUM_CTX", 8192)
    request_timeout: float = _env_float("SOC_REQUEST_TIMEOUT", 180.0)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("SOC_CHUNK_OVERLAP must be smaller than SOC_CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("SOC_TOP_K must be >= 1")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current environment.

        Defaults already read the environment at class-definition time; this
        classmethod simply re-evaluates them so changes made after import (e.g.
        in tests) are picked up.
        """
        return cls(
            ollama_host=_env_str("OLLAMA_HOST", "http://localhost:11434"),
            llm_model=_env_str("SOC_LLM_MODEL", "qwen2.5:14b"),
            embedding_model=_env_str("SOC_EMBEDDING_MODEL", "nomic-embed-text"),
            knowledge_base_dir=Path(_env_str("SOC_KB_DIR", str(_DEFAULT_KB_DIR))),
            chunk_size=_env_int("SOC_CHUNK_SIZE", 800),
            chunk_overlap=_env_int("SOC_CHUNK_OVERLAP", 120),
            top_k=_env_int("SOC_TOP_K", 4),
            temperature=_env_float("SOC_TEMPERATURE", 0.1),
            num_ctx=_env_int("SOC_NUM_CTX", 8192),
            request_timeout=_env_float("SOC_REQUEST_TIMEOUT", 180.0),
        )
