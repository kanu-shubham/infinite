"""
FAISS-backed Vector Store

Replaces the NumPy in-memory store for production use.

Why FAISS over NumPy?
---------------------
NumPy:  O(n) exact scan — fine up to ~100k chunks
FAISS:  approximate nearest neighbour — handles millions of chunks,
        10x-100x faster at scale, GPU support available

Index type used: IndexFlatIP (exact inner product search on normalised
vectors = exact cosine similarity). For >1M vectors swap to IndexIVFFlat
or IndexHNSWFlat for sub-linear query time.

Persistence
-----------
The index is saved to disk via faiss.write_index / read_index so it
survives process restarts.  Chunk metadata (content, doc info) is
stored alongside as a pickle file since FAISS only stores raw vectors.

ACL + Metadata filtering
-------------------------
Filtering happens BEFORE ranking (pre-filter), not after.
Two filter types:
  - ACL:      only return chunks the user is permitted to read
  - Metadata: filter by date, author, source, department etc.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import faiss
import numpy as np

from src.acl import UserContext, can_access
from src.documents import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------

@dataclass
class MetadataFilter:
    """
    A single filter predicate on chunk metadata.

    Examples
    --------
    MetadataFilter("source", "eq", "confluence")
    MetadataFilter("created_at", "gte", "2024-07-01")
    MetadataFilter("author", "in", ["alice", "bob"])
    MetadataFilter("title", "contains", "pricing")
    """
    field: str
    op: str       # "eq" | "gte" | "lte" | "in" | "contains"
    value: Any

    def matches(self, metadata: dict) -> bool:
        val = metadata.get(self.field)
        if val is None:
            return False
        if self.op == "eq":
            return str(val) == str(self.value)
        if self.op == "gte":
            return str(val) >= str(self.value)
        if self.op == "lte":
            return str(val) <= str(self.value)
        if self.op == "in":
            return val in self.value
        if self.op == "contains":
            return self.value.lower() in str(val).lower()
        return False


# ---------------------------------------------------------------------------
# FAISS store
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """
    Production vector store backed by FAISS with ACL and metadata filtering.

    Parameters
    ----------
    dim         : embedding dimension (must match your embedder)
    index_path  : directory to save/load index files (None = in-memory only)
    """

    def __init__(self, dim: int = 128, index_path: Optional[str] = None):
        self._dim = dim
        self._index_path = index_path
        self._chunks: List[Chunk] = []
        # IndexFlatIP: exact inner product (= cosine for normalised vectors)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with embeddings to the index."""
        if not chunks:
            return

        vecs = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding.")
            vec = chunk.embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vecs.append(vec)
            self._chunks.append(chunk)

        matrix = np.stack(vecs, axis=0)
        self._index.add(matrix)

    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """
        Remove chunks by ID.

        FAISS IndexFlatIP does not support deletion natively.
        We rebuild the index without the removed chunks.
        This is O(n) — for high-deletion workloads use IndexIDMap.
        """
        id_set = set(chunk_ids)
        keep = [(c, c.embedding) for c in self._chunks if c.chunk_id not in id_set]
        if len(keep) == len(self._chunks):
            return  # nothing to remove

        self._chunks = []
        self._index = faiss.IndexFlatIP(self._dim)
        if keep:
            chunks, embeddings = zip(*keep)
            self._chunks = list(chunks)
            matrix = np.stack([e.astype(np.float32) for e in embeddings])
            # Re-normalise
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms > 0, norms, 1)
            self._index.add(matrix)

    def clear(self) -> None:
        self._chunks = []
        self._index = faiss.IndexFlatIP(self._dim)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        min_score: float = 0.0,
        user: Optional[UserContext] = None,
        filters: Optional[List[MetadataFilter]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks with optional ACL and metadata pre-filtering.

        Parameters
        ----------
        query_vec : L2-normalised float32 query vector
        k         : number of results
        min_score : minimum similarity threshold
        user      : if provided, only return chunks this user can read
        filters   : additional metadata filter predicates (all must match)
        """
        if self._index.ntotal == 0:
            return []

        # Build permitted index mask
        permitted = self._permitted_indices(user, filters)
        if not permitted:
            return []

        # FAISS searches all vectors; we then filter by permitted mask.
        # Over-fetch to ensure we have k results after filtering.
        fetch_k = min(len(self._chunks), max(k * 10, k + len(self._chunks) - len(permitted)))

        q = query_vec.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores, indices = self._index.search(q.reshape(1, -1), fetch_k)
        scores, indices = scores[0], indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            if idx not in permitted:
                continue
            if float(score) < min_score:
                continue
            results.append(RetrievalResult(chunk=self._chunks[idx], score=float(score)))
            if len(results) >= k:
                break

        return results

    def _permitted_indices(
        self,
        user: Optional[UserContext],
        filters: Optional[List[MetadataFilter]],
    ) -> set:
        """Return set of chunk indices the user is allowed to see."""
        permitted = set()
        for i, chunk in enumerate(self._chunks):
            # ACL check
            if user is not None:
                perms = chunk.metadata.get("permissions", [])
                if not can_access(user, perms):
                    continue
            # Metadata filter check (all filters must match)
            if filters:
                if not all(f.matches(chunk.metadata) for f in filters):
                    continue
            permitted.add(i)
        return permitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save index and chunk metadata to disk."""
        if not self._index_path:
            raise ValueError("index_path not set — cannot save")
        os.makedirs(self._index_path, exist_ok=True)
        faiss.write_index(self._index,
                          os.path.join(self._index_path, "index.faiss"))
        with open(os.path.join(self._index_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> None:
        """Load index and chunk metadata from disk."""
        if not self._index_path:
            raise ValueError("index_path not set — cannot load")
        idx_file = os.path.join(self._index_path, "index.faiss")
        chunks_file = os.path.join(self._index_path, "chunks.pkl")
        if not os.path.exists(idx_file):
            return  # nothing saved yet
        self._index = faiss.read_index(idx_file)
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)

    # ------------------------------------------------------------------
    # Compatibility shim (same API as InMemoryVectorStore)
    # ------------------------------------------------------------------

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)
