"""Tests for knowledge-base loading and chunking."""

from __future__ import annotations

import pytest

from soc_summarizer.knowledge_base import (
    Document,
    chunk_document,
    chunk_documents,
    load_documents,
)
from tests.conftest import KB_DIR


def test_load_documents_reads_all_supported_files():
    docs = load_documents(KB_DIR)
    sources = {d.source for d in docs}
    assert sources == {
        "asset_classification.md",
        "brute_force_playbook.md",
        "fp_patterns.txt",
        "tor_exit_nodes.md",
    }
    assert all(d.text for d in docs)


def test_load_documents_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_load_documents_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        load_documents(tmp_path)


def test_chunking_preserves_source_and_indices():
    docs = load_documents(KB_DIR)
    chunks = chunk_documents(docs, chunk_size=800, overlap=120)

    assert len(chunks) >= len(docs)
    assert all(c.text.strip() for c in chunks)
    assert all(c.source in {d.source for d in docs} for c in chunks)
    # Global indices are contiguous and ordered.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_markdown_headings_are_captured():
    docs = load_documents(KB_DIR)
    playbook = next(d for d in docs if d.source == "brute_force_playbook.md")
    chunks = chunk_document(playbook, chunk_size=800, overlap=120)
    headings = {c.heading for c in chunks if c.heading}
    # The playbook has clear section headings that should survive chunking.
    assert any("Brute Force" in h for h in headings)
    # The heading is folded into the embedding text for better retrieval.
    assert any(c.heading and c.heading in c.embedding_text() for c in chunks)


def test_long_section_is_split_with_overlap():
    long_body = "\n\n".join(f"Sentence number {i} about authentication failures." for i in range(60))
    doc = Document(source="big.md", text=f"# Title\n\n{long_body}")
    chunks = chunk_document(doc, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 60 + 50 for c in chunks)  # size + overlap + slack
