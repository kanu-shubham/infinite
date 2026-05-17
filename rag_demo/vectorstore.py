"""
vectorstore.py — Store chunk embeddings and find the most similar ones to a query.

Analogy: a library catalogue where every book has a "meaning fingerprint" (embedding).
When you search, we compare your query's fingerprint to every book's fingerprint
and return the closest ones.
"""
import numpy as np
from dataclasses import dataclass
from documents import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float    # cosine similarity, 0–1, higher = more relevant


class VectorStore:
    """
    In-memory vector store backed by a NumPy matrix.

    Internal state:
        _chunks  : list of Chunk objects in insertion order
        _matrix  : 2D NumPy array, shape (N, dim)
                   row i = embedding for _chunks[i]
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None   # built lazily

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Store chunks alongside their embeddings.
        Each chunk must have chunk.embedding set before calling this.
        """
        if not chunks:
            return

        new_embeddings = np.array([c.embedding for c in chunks])
        # shape: (len(chunks), dim)

        self._chunks.extend(chunks)

        if self._matrix is None:
            # First batch — just store the matrix directly
            self._matrix = new_embeddings
        else:
            # Stack vertically: existing rows on top, new rows below
            # np.vstack([[a,b],[c,d]], [[e,f]]) → [[a,b],[c,d],[e,f]]
            self._matrix = np.vstack([self._matrix, new_embeddings])

    def remove_chunks(self, chunk_ids: set[str]) -> None:
        """Delete specific chunks (used during re-indexing)."""
        keep = [i for i, c in enumerate(self._chunks) if c.chunk_id not in chunk_ids]
        self._chunks = [self._chunks[i] for i in keep]
        if keep:
            self._matrix = self._matrix[keep]
        else:
            self._matrix = None

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        """
        Find the top_k most similar chunks to the query embedding.

        Math:
            scores = matrix @ query        ← dot product of every row with query
            Since both matrix rows and query are L2-normalised (unit vectors):
                dot product = cosine_similarity
            So scores[i] ∈ [-1, 1], higher = more similar.

            argpartition(scores, -top_k)[-top_k:]
            ← finds the indices of the top_k largest values WITHOUT fully sorting.
              This is O(N) instead of O(N log N) — important for millions of chunks.
        """
        if self._matrix is None or len(self._chunks) == 0:
            return []

        # matrix @ query_embedding
        # _matrix shape: (N, dim)
        # query_embedding shape: (dim,)
        # result shape: (N,)  — one score per chunk
        scores = self._matrix @ query_embedding

        k = min(top_k, len(self._chunks))

        # argpartition: indices of the k largest values (unordered)
        top_indices = np.argpartition(scores, -k)[-k:]

        # Now sort just those k indices by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)
