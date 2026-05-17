"""
documents.py — Core data models for the RAG system.
Every piece of text flowing through the pipeline is either a Document or a Chunk.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Document:
    """
    A raw document as it arrives from a connector (Confluence, SharePoint, etc.)
    """
    content: str            # the full text
    title: str = ""         # document title
    source: str = ""        # where it came from: "confluence/hr-policy"
    metadata: dict = field(default_factory=dict)   # any extra fields
    doc_id: str = field(init=False)                # computed, not passed in

    def __post_init__(self):
        # Build a fingerprint from the stable identity fields.
        # content[:200] — first 200 chars so two docs with same title/source
        # but different content get different IDs.
        fingerprint = f"{self.title}::{self.source}::{self.content[:200]}"
        # MD5 gives a 32-char hex string. We take the first 12 chars —
        # collision probability is negligible for millions of docs.
        self.doc_id = hashlib.md5(fingerprint.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """
    A piece of a Document after splitting. This is what gets embedded and stored.
    """
    content: str
    doc_id: str
    doc_title: str
    doc_source: str
    start_char: int        # character offset in the original document
    end_char: int
    chunk_index: int       # 0-based position within the document
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # chunk_id is deterministic from doc_id + position.
        # Same chunk re-indexed → same ID → upsert not duplicate.
        fingerprint = f"{self.doc_id}::{self.start_char}::{self.end_char}"
        self.chunk_id = hashlib.md5(fingerprint.encode()).hexdigest()[:12]


def chunk_document(doc: Document, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """
    Split a Document into overlapping Chunks of ~chunk_size characters.
    Overlap ensures a sentence that straddles a boundary appears in both chunks,
    so retrieval never misses context at the edge.
    """
    text = doc.content
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Don't cut in the middle of a word — walk back to the last space.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary != -1:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                content=chunk_text,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_source=doc.source,
                start_char=start,
                end_char=end,
                chunk_index=idx,
                metadata=doc.metadata.copy(),
            ))
            idx += 1

        # Move forward by chunk_size - overlap so consecutive chunks share
        # 'overlap' characters.
        start = end - overlap if end - overlap > start else end

    return chunks
