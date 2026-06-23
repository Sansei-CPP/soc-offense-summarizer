"""Tests for the in-memory cosine-similarity vector store."""

from __future__ import annotations

import pytest

from soc_summarizer.knowledge_base import Chunk
from soc_summarizer.vector_store import InMemoryVectorStore


def _chunk(name: str) -> Chunk:
    return Chunk(source=name, text=name, index=0)


def test_search_ranks_by_cosine_similarity():
    store = InMemoryVectorStore()
    store.add(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    results = store.search([1.0, 0.0], k=3)

    assert [c.source for c, _ in results] == ["a", "c", "b"]
    # 'a' is identical direction -> cosine 1.0; 'b' is orthogonal -> 0.0.
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)
    assert results[-1][1] == pytest.approx(0.0, abs=1e-6)


def test_search_respects_k():
    store = InMemoryVectorStore()
    store.add([_chunk("a"), _chunk("b")], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store.search([1.0, 1.0], k=1)) == 1


def test_magnitude_does_not_affect_cosine():
    store = InMemoryVectorStore()
    store.add([_chunk("a")], [[3.0, 0.0]])  # scaled version of the query direction
    (_, score), = store.search([1.0, 0.0], k=1)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_empty_store_returns_nothing():
    assert InMemoryVectorStore().search([1.0, 0.0], k=3) == []


def test_dimension_mismatch_raises():
    store = InMemoryVectorStore()
    store.add([_chunk("a")], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.add([_chunk("b")], [[1.0, 0.0, 0.0]])


def test_length_and_dim_bookkeeping():
    store = InMemoryVectorStore()
    assert len(store) == 0 and store.dim is None
    store.add([_chunk("a"), _chunk("b")], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store) == 2 and store.dim == 2
