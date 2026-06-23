"""Semantic retriever: chunk + embed the knowledge base, then search it."""

from __future__ import annotations

from pathlib import Path

from .embeddings import Embedder
from .knowledge_base import chunk_documents, load_documents
from .models import RetrievedChunk
from .vector_store import InMemoryVectorStore


class Retriever:
    """Builds the vector index at startup and answers semantic queries."""

    def __init__(self, embedder: Embedder, store: InMemoryVectorStore | None = None) -> None:
        self._embedder = embedder
        self._store = store or InMemoryVectorStore()
        self._chunk_count = 0
        self._doc_count = 0

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def document_count(self) -> int:
        return self._doc_count

    def build(self, knowledge_base_dir: Path, chunk_size: int, chunk_overlap: int) -> "Retriever":
        """Load documents from disk, chunk, embed, and populate the store."""
        documents = load_documents(knowledge_base_dir)
        chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        embeddings = self._embedder.embed_documents([c.embedding_text() for c in chunks])
        self._store.add(chunks, embeddings)
        self._doc_count = len(documents)
        self._chunk_count = len(chunks)
        return self

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed_query(query)
        hits = self._store.search(query_embedding, k)
        return [
            RetrievedChunk(
                source=chunk.source,
                heading=chunk.heading,
                text=chunk.text,
                score=round(score, 4),
            )
            for chunk, score in hits
        ]
