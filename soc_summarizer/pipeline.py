"""The end-to-end offense-summarization pipeline.

``OffenseSummarizer`` wires together the retriever (embeddings + vector store)
and the LLM. The knowledge base is loaded, chunked, embedded, and indexed when
the summarizer is constructed — i.e. at startup — satisfying the requirement to
populate the vector store from the provided documents rather than hardcoding it.

The embedder and LLM are injectable so the whole pipeline can be unit-tested
without a running Ollama server.
"""

from __future__ import annotations

import json
from typing import Union

from pydantic import ValidationError

from .config import Settings
from .embeddings import Embedder
from .llm import LLMClient
from .models import AnalysisResult, Offense, OffenseAnalysis, RetrievedChunk
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever

OffenseInput = Union[Offense, dict]


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if a model wrapped its JSON in them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return stripped.strip()


class AnalysisError(RuntimeError):
    """Raised when the LLM output cannot be parsed into the expected schema."""


class OffenseSummarizer:
    """Turns a raw offense into a structured, analyst-ready AI summary."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()

        # Lazily construct production backends only if not injected, so importing
        # / testing the pipeline never requires the optional ``ollama`` package.
        if embedder is None:
            from .embeddings import OllamaEmbedder

            embedder = OllamaEmbedder(
                model=self.settings.embedding_model,
                host=self.settings.ollama_host,
                timeout=self.settings.request_timeout,
            )
        if llm is None:
            from .llm import OllamaLLM

            llm = OllamaLLM(
                model=self.settings.llm_model,
                host=self.settings.ollama_host,
                temperature=self.settings.temperature,
                num_ctx=self.settings.num_ctx,
                timeout=self.settings.request_timeout,
            )

        self._llm = llm
        # Build (load -> chunk -> embed -> index) the knowledge base at startup.
        self._retriever = Retriever(embedder).build(
            knowledge_base_dir=self.settings.knowledge_base_dir,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # -- introspection --------------------------------------------------------

    @property
    def document_count(self) -> int:
        return self._retriever.document_count

    @property
    def chunk_count(self) -> int:
        return self._retriever.chunk_count

    # -- core -----------------------------------------------------------------

    @staticmethod
    def _coerce(offense: OffenseInput) -> Offense:
        return offense if isinstance(offense, Offense) else Offense.model_validate(offense)

    def retrieve(self, offense: OffenseInput) -> tuple[str, list[RetrievedChunk]]:
        """Run only the retrieval stage (used by `--dry-run`)."""
        parsed = self._coerce(offense)
        query = parsed.to_retrieval_query()
        chunks = self._retriever.retrieve(query, self.settings.top_k)
        return query, chunks

    def analyze(self, offense: OffenseInput) -> AnalysisResult:
        """Full pipeline: retrieve context, prompt the LLM, validate the output."""
        parsed = self._coerce(offense)
        query, chunks = self.retrieve(parsed)

        user_prompt = build_user_prompt(parsed, chunks)
        schema = OffenseAnalysis.model_json_schema()
        raw = self._llm.generate_json(SYSTEM_PROMPT, user_prompt, schema)

        try:
            analysis = OffenseAnalysis.model_validate_json(_strip_code_fences(raw))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise AnalysisError(
                "LLM did not return a valid OffenseAnalysis object.\n"
                f"Raw output:\n{raw}"
            ) from exc

        return AnalysisResult(
            analysis=analysis,
            retrieval_query=query,
            retrieved_chunks=chunks,
            llm_model=self.settings.llm_model,
            embedding_model=self.settings.embedding_model,
        )
