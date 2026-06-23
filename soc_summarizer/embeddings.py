"""Embedding backends.

The ``Embedder`` protocol decouples the rest of the pipeline from any specific
provider. ``OllamaEmbedder`` is the production implementation, talking to a local
Ollama server. Asymmetric retrieval prefixes (``search_document:`` /
``search_query:``) are applied automatically for ``nomic-embed-text``, which was
trained with them — this measurably improves retrieval quality.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into vectors for the vector store."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def _extract_embeddings(response: Any) -> list[list[float]]:
    """Read the embedding matrix from an Ollama ``embed`` response.

    Handles both the typed response object (recent ollama-python) and the plain
    dict returned by older versions.
    """
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is None:
        raise RuntimeError(f"Unexpected Ollama embed response: {response!r}")
    return [list(vector) for vector in embeddings]


class OllamaEmbedder:
    """Embeds text via a local Ollama server."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        use_task_prefixes: bool | None = None,
    ) -> None:
        # Imported lazily so the package (and its tests, which use fakes) can be
        # imported without the optional ``ollama`` dependency installed.
        from ollama import Client

        self.model = model
        self._client = Client(host=host, timeout=timeout)
        # nomic-embed-* models expect asymmetric task prefixes; enable by default
        # when the model name looks like nomic, unless explicitly overridden.
        if use_task_prefixes is None:
            use_task_prefixes = "nomic" in model.lower()
        self._use_task_prefixes = use_task_prefixes

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = self._client.embed(model=self.model, input=inputs)
        vectors = _extract_embeddings(response)
        if len(vectors) != len(inputs):
            raise RuntimeError(
                f"Ollama returned {len(vectors)} embeddings for {len(inputs)} inputs"
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [
            f"search_document: {t}" if self._use_task_prefixes else t for t in texts
        ]
        return self._embed(prepared)

    def embed_query(self, text: str) -> list[float]:
        prepared = f"search_query: {text}" if self._use_task_prefixes else text
        return self._embed([prepared])[0]
