"""LLM backends.

``LLMClient`` is the protocol the pipeline depends on. ``OllamaLLM`` is the
production implementation: it calls a local Ollama chat model and uses Ollama's
structured-output feature (``format`` = a JSON schema) so the model is
constrained at decode time to emit JSON matching our ``OffenseAnalysis`` schema.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can answer a system+user prompt as schema-constrained JSON."""

    def generate_json(self, system: str, user: str, schema: dict) -> str:
        ...


def _extract_content(response: Any) -> str:
    """Read assistant text from an Ollama chat response (object or dict)."""
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    raise RuntimeError(f"Unexpected Ollama chat response: {response!r}")


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


def _looks_like_connection_error(exc: Exception) -> bool:
    """Heuristically detect a 'server not reachable' error.

    Ollama uses httpx under the hood, whose connection errors do not subclass
    the builtin ``ConnectionError``; match on the exception family/message
    instead of importing httpx eagerly.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "connect" in name
        or "connection refused" in text
        or "max retries" in text
        or "failed to establish" in text
    )


class OllamaLLM:
    """Generates structured JSON via a local Ollama chat model."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        timeout: float = 180.0,
    ) -> None:
        from ollama import Client

        self.model = model
        self._host = host
        self._client = Client(host=host, timeout=timeout)
        self._options = {"temperature": temperature, "num_ctx": num_ctx}

    def generate_json(self, system: str, user: str, schema: dict) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema,  # constrained decoding against the JSON schema
                options=self._options,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised unless it's a connection error
            if _looks_like_connection_error(exc):
                raise OllamaConnectionError(
                    f"Could not reach Ollama at {self._host!r}. "
                    "Is 'ollama serve' running?"
                ) from exc
            raise
        return _extract_content(response).strip()
