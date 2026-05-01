"""
Incremental Index Manager

Handles the operational lifecycle of documents in the index:
  - Add new documents
  - Update changed documents (delete old chunks, insert new)
  - Delete removed documents
  - Track which chunks belong to which document

Without this, re-indexing always wipes and rebuilds everything from
scratch — which is too slow and expensive for large corpora where
only a small fraction of documents change each day.

Usage
-----
  manager = IncrementalIndexManager(store, embedder, chunk_size=512)
  manager.upsert(doc)        # add or update a document
  manager.delete(doc_id)     # remove a document
  manager.batch_upsert(docs) # bulk update
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.documents import Document, Chunk, chunk_document
from src.embeddings.base import BaseEmbedder
from src.config import config


@dataclass
class IndexStats:
    total_docs: int = 0
    total_chunks: int = 0
    last_updated: float = field(default_factory=time.time)


class IncrementalIndexManager:
    """
    Manages document-level upsert/delete on top of any vector store.

    Works with both InMemoryVectorStore and FAISSVectorStore — anything
    that implements add_chunks() and remove_chunks().

    Parameters
    ----------
    store    : vector store instance (must support add_chunks + remove_chunks)
    embedder : fitted BaseEmbedder
    chunk_size / chunk_overlap : chunking parameters
    """

    def __init__(
        self,
        store,
        embedder: BaseEmbedder,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size or config.chunk_size
        self._chunk_overlap = chunk_overlap or config.chunk_overlap

        # doc_id → list of chunk_ids currently indexed for that doc
        self._doc_chunks: Dict[str, List[str]] = {}
        # doc_id → Document (for reference / debugging)
        self._docs: Dict[str, Document] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, doc: Document) -> List[Chunk]:
        """
        Add or update a document in the index.

        If the document was previously indexed, its old chunks are
        removed before new chunks are inserted.  This handles both
        new documents and edited documents identically.

        Returns the list of new Chunk objects that were indexed.
        """
        # Remove old version if present
        if doc.doc_id in self._doc_chunks:
            self.delete(doc.doc_id)

        # Chunk
        chunks = chunk_document(doc, self._chunk_size, self._chunk_overlap)
        if not chunks:
            return []

        # Embed
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Store
        self._store.add_chunks(chunks)

        # Track
        self._doc_chunks[doc.doc_id] = [c.chunk_id for c in chunks]
        self._docs[doc.doc_id] = doc

        return chunks

    def delete(self, doc_id: str) -> None:
        """
        Remove all chunks belonging to *doc_id* from the index.
        No-op if the document is not indexed.
        """
        chunk_ids = self._doc_chunks.pop(doc_id, [])
        self._docs.pop(doc_id, None)
        if chunk_ids:
            self._store.remove_chunks(chunk_ids)

    def batch_upsert(self, docs: List[Document]) -> None:
        """
        Upsert many documents.

        For new corpora, batch_upsert is more efficient than calling
        upsert() in a loop because it embeds all texts in one call.
        For mixed new/updated docs the gain is smaller.
        """
        # Delete old versions first
        for doc in docs:
            if doc.doc_id in self._doc_chunks:
                self.delete(doc.doc_id)

        # Chunk all docs
        all_chunks: List[Chunk] = []
        doc_chunk_map: Dict[str, List[Chunk]] = {}
        for doc in docs:
            chunks = chunk_document(doc, self._chunk_size, self._chunk_overlap)
            doc_chunk_map[doc.doc_id] = chunks
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # Embed all at once
        texts = [c.content for c in all_chunks]
        embeddings = self._embedder.embed_texts(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # Store
        self._store.add_chunks(all_chunks)

        # Track
        for doc in docs:
            chunks = doc_chunk_map[doc.doc_id]
            self._doc_chunks[doc.doc_id] = [c.chunk_id for c in chunks]
            self._docs[doc.doc_id] = doc

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def stats(self) -> IndexStats:
        return IndexStats(
            total_docs=len(self._doc_chunks),
            total_chunks=sum(len(v) for v in self._doc_chunks.values()),
        )

    def is_indexed(self, doc_id: str) -> bool:
        return doc_id in self._doc_chunks

    def chunk_ids_for(self, doc_id: str) -> List[str]:
        return list(self._doc_chunks.get(doc_id, []))
