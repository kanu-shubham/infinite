"""
ANN Index — wraps FAISS IndexFlatIP for exact inner-product search.

Since both user and item embeddings are L2-normalised, inner product
equals cosine similarity.  In production you would swap IndexFlatIP for
IndexIVFFlat or IndexHNSWFlat to get sub-linear retrieval time at the
cost of a small approximation error.

Interface deliberately matches what a real FAISS serving layer would
expose, so the pipeline code does not need to change if you swap backends.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss                    # pip install faiss-cpu
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class ANNIndex:
    """
    Wraps a FAISS flat inner-product index with product-ID bookkeeping.

    Usage
    -----
    index = ANNIndex(embed_dim=128)
    index.build(item_embeddings, product_ids)
    scores, ids = index.search(user_embedding, top_k=500)
    index.save("artifacts/ann_index.pkl")
    index2 = ANNIndex.load("artifacts/ann_index.pkl")
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim   = embed_dim
        self._index      = None
        self._product_ids: List[str] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, product_ids: List[str]) -> None:
        """
        Index a catalogue of item embeddings.

        Parameters
        ----------
        embeddings   : float32 (N, embed_dim), must be unit-normalised.
        product_ids  : list of N product ID strings (positional, same order).
        """
        assert embeddings.shape[1] == self.embed_dim, (
            f"Expected embed_dim={self.embed_dim}, got {embeddings.shape[1]}"
        )
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb)          # ensure unit norm before indexing

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.embed_dim)
            self._index.add(emb)
        else:
            # Fallback: store the matrix and do exact numpy dot-product search
            self._index = emb.copy()

        self._product_ids = list(product_ids)
        print(f"ANNIndex: indexed {len(product_ids)} products (dim={self.embed_dim})")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int = 500,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve the top-k nearest products for a query user embedding.

        Parameters
        ----------
        query  : float32 (embed_dim,)  — unit-normalised user embedding.
        top_k  : number of candidates to return.

        Returns
        -------
        scores      : float32 (k,)  cosine similarities in [-1, 1]
        product_ids : list of k product ID strings
        """
        assert self._index is not None, "Call build() or load() first."
        q = np.ascontiguousarray(query[None], dtype=np.float32)    # (1, D)
        faiss.normalize_L2(q)

        k = min(top_k, len(self._product_ids))

        if _FAISS_AVAILABLE:
            scores_2d, indices_2d = self._index.search(q, k)
            scores  = scores_2d[0]
            indices = indices_2d[0]
        else:
            # Exact numpy cosine search
            sims    = (self._index @ q[0]).ravel()
            indices = np.argsort(sims)[::-1][:k]
            scores  = sims[indices]

        pids = [self._product_ids[i] for i in indices]
        return scores, pids

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _FAISS_AVAILABLE:
            # Serialise the FAISS index to bytes, then pickle everything
            index_bytes = faiss.serialize_index(self._index).tobytes()
            payload = {"embed_dim": self.embed_dim, "product_ids": self._product_ids,
                       "index_bytes": index_bytes, "backend": "faiss"}
        else:
            payload = {"embed_dim": self.embed_dim, "product_ids": self._product_ids,
                       "matrix": self._index, "backend": "numpy"}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"ANNIndex saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ANNIndex":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(embed_dim=payload["embed_dim"])
        obj._product_ids = payload["product_ids"]
        if payload.get("backend") == "faiss" and _FAISS_AVAILABLE:
            buf         = np.frombuffer(payload["index_bytes"], dtype=np.uint8)
            obj._index  = faiss.deserialize_index(buf)
        else:
            obj._index = payload.get("matrix")
        print(f"ANNIndex loaded ← {path}  ({len(obj._product_ids)} products)")
        return obj
