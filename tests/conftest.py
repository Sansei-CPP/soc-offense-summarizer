"""Shared test fixtures and fakes.

The fakes let the entire pipeline run without a live Ollama server:

* ``LexicalEmbedder`` is a deterministic hashing bag-of-words embedder. Cosine
  similarity over its vectors reflects token overlap, so it produces *meaningful*
  retrieval rankings in tests (relevant docs really do rank higher) while staying
  dependency-free and reproducible.
* ``FakeLLM`` returns a fixed, schema-valid ``OffenseAnalysis`` JSON document.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import pytest

from soc_summarizer.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "data" / "knowledge_base"
SAMPLE_OFFENSE_PATH = REPO_ROOT / "examples" / "offense_sample.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LexicalEmbedder:
    """Deterministic bag-of-words hashing embedder (no external deps)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            # Stable bucket via md5 so results don't depend on PYTHONHASHSEED.
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


_FAKE_ANALYSIS = {
    "summary": "External source brute-forced SSH and obtained one successful login.",
    "classification": "TP",
    "classification_rationale": "A successful authentication followed many failures against a production host.",
    "recommended_action": "Escalate per the SSH/RDP Brute Force playbook and review the deploy-svc account.",
    "confidence": "High",
    "confidence_pct": 90,
    "key_indicators": [
        "successful auth among brute-force failures",
        "source is a Tor exit node",
        "target is a production auth server",
    ],
    "referenced_sources": ["brute_force_playbook.md", "tor_exit_nodes.md"],
}


class FakeLLM:
    """Returns a fixed, schema-valid OffenseAnalysis JSON document."""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or _FAKE_ANALYSIS
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_schema: dict | None = None

    def generate_json(self, system: str, user: str, schema: dict) -> str:
        self.last_system = system
        self.last_user = user
        self.last_schema = schema
        return json.dumps(self.payload)


@pytest.fixture
def settings() -> Settings:
    return Settings(knowledge_base_dir=KB_DIR, top_k=4)


@pytest.fixture
def lexical_embedder() -> LexicalEmbedder:
    return LexicalEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_offense() -> dict:
    return json.loads(SAMPLE_OFFENSE_PATH.read_text(encoding="utf-8"))
