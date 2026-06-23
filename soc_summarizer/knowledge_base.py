"""Knowledge-base loading and chunking.

Documents are loaded from a directory at startup (never hardcoded) and split
into overlapping, heading-aware chunks suitable for embedding. Markdown is split
on ``#``/``##`` headings so each playbook section stays semantically intact; the
heading is carried into the chunk text and metadata to preserve context. Long
sections are further split by size with overlap. Plain-text files are packed
paragraph-by-paragraph into size-bounded chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Document:
    """A raw knowledge-base document."""

    source: str  # file name, e.g. "brute_force_playbook.md"
    text: str


@dataclass(frozen=True)
class Chunk:
    """A unit of knowledge-base text ready for embedding."""

    source: str
    text: str
    index: int
    heading: str | None = None
    metadata: dict = field(default_factory=dict)

    def embedding_text(self) -> str:
        """Text to embed: prepend the heading so the section topic is encoded."""
        if self.heading:
            return f"{self.heading}\n{self.text}"
        return self.text


def load_documents(directory: Path) -> list[Document]:
    """Load all supported documents from ``directory`` (sorted for determinism)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(Document(source=path.name, text=text))

    if not documents:
        raise ValueError(
            f"No knowledge base documents ({', '.join(sorted(SUPPORTED_SUFFIXES))}) "
            f"found in {directory}"
        )
    return documents


def _split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split markdown into (heading, body) sections at ATX headings.

    Content before the first heading is returned with a ``None`` heading.
    """
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or current_heading:
            sections.append((current_heading, body))

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def _split_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into <= chunk_size pieces with ``overlap`` characters of
    carry-over, breaking on paragraph/sentence boundaries where possible."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # Prefer paragraph boundaries, then sentence boundaries.
    units = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    current = ""

    def emit(piece: str) -> None:
        piece = piece.strip()
        if piece:
            pieces.append(piece)

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > chunk_size:
            # Hard-wrap an oversized paragraph on sentence boundaries.
            if current:
                emit(current)
                current = ""
            for sentence in re.split(r"(?<=[.!?])\s+", unit):
                if len(current) + len(sentence) + 1 <= chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    emit(current)
                    current = sentence
            continue
        if len(current) + len(unit) + 2 <= chunk_size:
            current = f"{current}\n\n{unit}".strip()
        else:
            emit(current)
            current = unit
    emit(current)

    if overlap <= 0 or len(pieces) <= 1:
        return pieces

    # Add character overlap between consecutive pieces for retrieval continuity.
    overlapped: list[str] = [pieces[0]]
    for prev, piece in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        overlapped.append(f"{tail}\n{piece}".strip())
    return overlapped


def chunk_document(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a single document into heading-aware, size-bounded pieces."""
    chunks: list[Chunk] = []
    sections = _split_markdown_sections(document.text)

    for heading, body in sections:
        for piece in _split_by_size(body, chunk_size, overlap):
            chunks.append(
                Chunk(
                    source=document.source,
                    text=piece,
                    index=len(chunks),
                    heading=heading,
                    metadata={"source": document.source, "heading": heading},
                )
            )
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Chunk every document. Indices are global across the whole corpus."""
    chunks: list[Chunk] = []
    for document in documents:
        for chunk in chunk_document(document, chunk_size, overlap):
            chunks.append(
                Chunk(
                    source=chunk.source,
                    text=chunk.text,
                    index=len(chunks),
                    heading=chunk.heading,
                    metadata=chunk.metadata,
                )
            )
    return chunks
