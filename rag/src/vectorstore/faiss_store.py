"""
FAISS-backed Vector Store

Replaces the NumPy in-memory store for production use.

Why FAISS over NumPy?
---------------------
NumPy:  O(n) exact scan — fine up to ~100k chunks
FAISS:  approximate nearest neighbour — handles millions of chunks,
        10x-100x faster at scale, GPU support available

Index types
-----------
Three index types are supported, selected by the `index_type` param:

  "flat"  (IndexFlatIP) — DEFAULT
    Exact brute-force cosine search.
    No training required, perfect recall.
    O(n) per query — fine for < 1M vectors.
    Use when: index fits in memory, recall is critical.

  "hnsw"  (IndexHNSWFlat) — RECOMMENDED for production
    Hierarchical Navigable Small World graph.
    Approximate nearest neighbour, no training required.
    O(log n) per query, tunable recall/speed trade-off via M
    (number of edges per node, default 32) and efSearch (beam width).
    Typical recall@10: 95-99%.
    Use when: low latency matters and you have > 100k vectors.
    Memory: ~(4 * dim + M * 8) bytes per vector.

  "ivf"  (IndexIVFFlat) — BEST for very large corpora
    Inverted File Index: clusters vectors into nlist Voronoi cells.
    At query time, searches only nprobe cells (default 8).
    REQUIRES training on a representative sample first.
    O(n/nlist * nprobe) per query — can be 100x faster than flat.
    Use when: corpus > 1M vectors and you can afford a training pass.

    Trade-off: nprobe ↑ = recall ↑ = latency ↑

    HNSW vs IVF:
    - HNSW: better recall at same speed, no training, uses more memory
    - IVF:  less memory, needs training, requires nprobe tuning

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
    index_type  : "flat" | "hnsw" | "ivf"  (see module docstring for trade-offs)
    hnsw_m      : HNSW connections per node (higher = better recall, more memory)
    ivf_nlist   : IVF number of Voronoi cells (sqrt(n) is a good default)
    ivf_nprobe  : IVF cells to search at query time (higher = better recall)
    """

    def __init__(
        self,
        dim: int = 128,
        index_path: Optional[str] = None,
        index_type: str = "flat",
        hnsw_m: int = 32,
        ivf_nlist: int = 100,
        ivf_nprobe: int = 8,
    ):
        self._dim = dim
        self._index_path = index_path
        self._index_type = index_type
        self._hnsw_m = hnsw_m
        self._ivf_nlist = ivf_nlist
        self._ivf_nprobe = ivf_nprobe
        self._chunks: List[Chunk] = []
        self._ivf_trained = False
        self._index = self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self):
        """Create the appropriate FAISS index based on index_type."""
        if self._index_type == "hnsw":
            # IndexHNSWFlat: graph-based ANN, no training needed
            index = faiss.IndexHNSWFlat(self._dim, self._hnsw_m,
                                        faiss.METRIC_INNER_PRODUCT)
            # efSearch controls query-time recall vs speed (default 16, raise for better recall)
            index.hnsw.efSearch = 64
            return index

        elif self._index_type == "ivf":
            # IndexIVFFlat: cluster-based ANN, requires training
            # Wrap in IndexFlatIP quantizer for the coarse quantiser step
            quantizer = faiss.IndexFlatIP(self._dim)
            index = faiss.IndexIVFFlat(
                quantizer, self._dim, self._ivf_nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.nprobe = self._ivf_nprobe
            return index

        else:  # "flat" (default)
            return faiss.IndexFlatIP(self._dim)

    def _ensure_ivf_trained(self, matrix: np.ndarray) -> None:
        """Train the IVF index if not yet trained. Called lazily on first add."""
        if self._index_type != "ivf" or self._ivf_trained:
            return
        if matrix.shape[0] < self._ivf_nlist:
            # Not enough vectors to train — fall back to flat for now
            self._index = faiss.IndexFlatIP(self._dim)
            self._index_type = "flat"
            return
        self._index.train(matrix)
        self._ivf_trained = True

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
        self._ensure_ivf_trained(matrix)
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
        self._ivf_trained = False
        self._index = self._build_index()
        if keep:
            chunks, embeddings = zip(*keep)
            self._chunks = list(chunks)
            matrix = np.stack([e.astype(np.float32) for e in embeddings])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms > 0, norms, 1)
            self._ensure_ivf_trained(matrix)
            self._index.add(matrix)

    def clear(self) -> None:
        self._chunks = []
        self._ivf_trained = False
        self._index = self._build_index()

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
