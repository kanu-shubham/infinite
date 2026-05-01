"""
In-memory vector store backed by a NumPy matrix.

All embeddings are stored in a single float32 matrix.  Retrieval is a
batched cosine-similarity dot-product (O(n) scan) which is fine up to
~100 k chunks on modern hardware.  For larger corpora, swap this for
FAISS (IndexFlatIP) or a managed vector DB (Pinecone, Weaviate, pgvector).

Because all vectors are L2-normalised at insertion time, cosine similarity
collapses to a plain dot product — fast with NumPy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.documents import Chunk, RetrievalResult


class InMemoryVectorStore:
    """
    A simple flat vector index.

    Usage
    -----
    store = InMemoryVectorStore()
    store.add_chunks(chunks)                  # add Chunk objects with .embedding set
    results = store.search(query_vec, k=5)    # returns List[RetrievalResult]
    """

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._matrix: Optional[np.ndarray] = None  # shape (n, dim)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add *chunks* to the index.  Each chunk must have `.embedding` set.

        Embeddings are L2-normalised on insertion so the stored matrix can
        be reused for dot-product similarity directly.
        """
        if not chunks:
            return

        new_vecs: List[np.ndarray] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has no embedding. "
                    "Call embedder.embed_texts() before adding to the store."
                )
            vec = chunk.embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            new_vecs.append(vec)
            self._chunks.append(chunk)

        new_matrix = np.stack(new_vecs, axis=0)  # (batch, dim)
        if self._matrix is None:
            self._matrix = new_matrix
        else:
            self._matrix = np.concatenate([self._matrix, new_matrix], axis=0)

    def remove_chunks(self, chunk_ids: list) -> None:
        """Remove chunks by ID, rebuilding the internal matrix."""
        id_set = set(chunk_ids)
        keep = [(c, c.embedding) for c in self._chunks if c.chunk_id not in id_set]
        self._chunks = []
        self._matrix = None
        if keep:
            chunks, embeddings = zip(*keep)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
            self.add_chunks(list(chunks))

    def clear(self) -> None:
        self._chunks = []
        self._matrix = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Return the top-k most similar chunks for *query_vec*.

        Parameters
        ----------
        query_vec  : L2-normalised float32 vector
        k          : number of results to return
        min_score  : minimum cosine similarity (hard filter)

        Returns
        -------
        List[RetrievalResult] sorted by descending score.
        An empty list is returned when the store is empty or all scores
        are below *min_score* (the "no result" case).
        """
        if self._matrix is None or len(self._chunks) == 0:
            return []

        q = query_vec.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Cosine similarity == dot product for normalised vectors
        scores: np.ndarray = self._matrix @ q  # (n,)

        k = min(k, len(self._chunks))
        # argpartition gives top-k in O(n) without full sort
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort desc

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score >= min_score:
                results.append(RetrievalResult(chunk=self._chunks[idx], score=score))

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks)
