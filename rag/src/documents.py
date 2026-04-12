"""
Document model and text-chunking utilities.

Concepts
--------
Document  — a raw source text with metadata (title, url, source, etc.)
Chunk     — a fixed-size window of a Document; the unit of retrieval.

Chunking strategy: sliding window with overlap.
  chunk_size    = max characters per chunk
  chunk_overlap = characters shared between consecutive chunks

Why overlap?  So that a sentence that falls on a boundary is fully
captured in at least one chunk, preventing "torn" context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A source document before chunking."""
    content: str
    title: str = ""
    source: str = ""          # URL, filename, or logical identifier
    metadata: dict = field(default_factory=dict)
    doc_id: str = field(init=False)

    def __post_init__(self):
        # Deterministic ID so re-indexing the same doc gives the same ID
        fingerprint = f"{self.title}::{self.source}::{self.content[:200]}"
        self.doc_id = hashlib.md5(fingerprint.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A retrieved-able slice of a Document."""
    content: str
    doc_id: str
    doc_title: str
    doc_source: str
    start_char: int
    end_char: int
    chunk_index: int          # position within its parent document
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    chunk_id: str = field(init=False)

    def __post_init__(self):
        fingerprint = f"{self.doc_id}::{self.start_char}::{self.end_char}"
        self.chunk_id = hashlib.md5(fingerprint.encode()).hexdigest()[:12]

    def citation(self) -> str:
        """Human-readable citation string."""
        parts = []
        if self.doc_title:
            parts.append(self.doc_title)
        if self.doc_source:
            parts.append(f"({self.doc_source})")
        parts.append(f"chars {self.start_char}–{self.end_char}")
        return " ".join(parts)


@dataclass
class RetrievalResult:
    """A chunk paired with its similarity score."""
    chunk: Chunk
    score: float

    def __repr__(self):
        return f"RetrievalResult(score={self.score:.3f}, chunk_id={self.chunk.chunk_id})"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _sentence_boundaries(text: str) -> List[int]:
    """Return character positions of sentence boundaries (after '.', '!', '?')."""
    positions = [0]
    for m in re.finditer(r'(?<=[.!?])\s+', text):
        positions.append(m.end())
    return positions


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Chunk]:
    """
    Split *doc* into overlapping character-level chunks aligned to sentence
    boundaries where possible.

    Algorithm
    ---------
    1. Find all sentence start positions.
    2. Walk forward: accumulate characters until chunk_size is reached.
    3. Back up to the nearest sentence boundary (greedy split).
    4. Next chunk starts chunk_overlap characters before the current end
       (again snapped to a sentence boundary).

    Returns
    -------
    List[Chunk] — ordered list of chunks for the document.
    """
    text = doc.content.strip()
    if not text:
        return []

    boundaries = _sentence_boundaries(text)
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # Tentative end
        end = min(start + chunk_size, len(text))

        # Snap end backward to the nearest sentence boundary that is > start
        if end < len(text):
            # Find the last boundary within [start+1, end]
            valid = [b for b in boundaries if start < b <= end]
            if valid:
                end = valid[-1]

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    content=chunk_text,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    doc_source=doc.source,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=dict(doc.metadata),
                )
            )
            chunk_index += 1

        if end >= len(text):
            break

        # Next chunk starts with overlap
        overlap_start = max(start + 1, end - chunk_overlap)
        # Snap overlap_start to the nearest boundary >= overlap_start
        valid_starts = [b for b in boundaries if b >= overlap_start]
        start = valid_starts[0] if valid_starts else end

    return chunks
