"""A minimal in-memory vector store.

Per the assignment, the store is rebuilt at startup from the knowledge-base
documents — no external vector database is needed for a corpus this size. It
holds L2-normalized embeddings in a NumPy matrix so similarity search is a
single matrix-vector product (cosine similarity).
"""

from __future__ import annotations

import numpy as np

from .knowledge_base import Chunk


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for empty vectors
    return matrix / norms


class InMemoryVectorStore:
    """Stores chunk embeddings and serves cosine-similarity nearest-neighbours."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int | None:
        return None if self._matrix is None else int(self._matrix.shape[1])

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        new_matrix = _normalize(np.asarray(embeddings, dtype=np.float32))
        if self._matrix is None:
            self._matrix = new_matrix
        else:
            if new_matrix.shape[1] != self._matrix.shape[1]:
                raise ValueError("embedding dimensionality mismatch")
            self._matrix = np.vstack([self._matrix, new_matrix])
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Return the top-``k`` (chunk, cosine_similarity) pairs, best first."""
        if self._matrix is None or not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        scores = self._matrix @ query  # cosine similarity (both sides normalized)
        k = min(k, len(self._chunks))
        # argpartition for the top-k, then sort just those by score descending.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]
