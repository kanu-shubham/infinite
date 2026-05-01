"""
Hybrid Search — Dense (semantic) + Sparse (BM25 keyword)

Why hybrid?
-----------
Dense retrieval (embeddings) excels at semantic similarity:
  query "car engine problems" → finds docs about "automobile motor issues"

Sparse retrieval (BM25) excels at exact matches:
  query "error code E-2847" → dense misses it (code is out-of-vocabulary)
                              BM25 finds it immediately

Combining both covers both failure modes.

BM25 (Best Match 25)
--------------------
An improved TF-IDF formula that adds:
  1. Term frequency saturation: the 5th occurrence of a word adds less
     signal than the 1st (controlled by parameter k1).
  2. Document length normalisation: long documents are penalised so they
     don't dominate just for being longer (controlled by parameter b).

  BM25(d, q) = Σ_t IDF(t) × [ TF(t,d) × (k1+1) ] / [ TF(t,d) + k1×(1 - b + b×|d|/avgdl) ]

Merging strategy
----------------
Dense and sparse scores live on different scales. We normalise each
list to [0, 1] then combine with a weighted sum:

  hybrid_score = alpha × dense_score + (1 - alpha) × bm25_score

where alpha=0.5 by default. Increase alpha for semantic queries,
decrease for keyword-heavy queries (product codes, IDs, etc.).

Alternatively (and often better in practice), use RRF to merge the
two ranked lists — no score normalisation needed.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.acl import UserContext
from src.documents import Chunk, RetrievalResult
from src.embeddings.base import BaseEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore, MetadataFilter


def _tokenise(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


def _rrf_merge(
    lists: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """Reciprocal Rank Fusion — same function used in query_decomp."""
    from collections import defaultdict
    scores: dict = defaultdict(float)
    best: dict = {}
    for ranked in lists:
        for rank, r in enumerate(ranked, start=1):
            cid = r.chunk.chunk_id
            scores[cid] += 1.0 / (k + rank)
            if cid not in best or r.score > best[cid].score:
                best[cid] = r
    fused = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [RetrievalResult(chunk=best[c].chunk, score=scores[c]) for c in fused]


class HybridRetriever:
    """
    Combines FAISS (dense) and BM25 (sparse) retrieval via RRF.

    Must be built after the corpus is indexed — call build_bm25() once
    the vector store is populated.

    Parameters
    ----------
    store    : FAISSVectorStore (or any store with .all_chunks())
    embedder : fitted BaseEmbedder
    alpha    : weight for dense vs sparse in score-fusion mode (unused when
               merge_strategy="rrf", which is the default)
    """

    def __init__(
        self,
        store: FAISSVectorStore,
        embedder: BaseEmbedder,
        alpha: float = 0.5,
    ):
        self._store = store
        self._embedder = embedder
        self.alpha = alpha

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunks: List[Chunk] = []

    def build_bm25(self) -> "HybridRetriever":
        """
        Build the BM25 index from all chunks currently in the vector store.
        Call once after indexing, and again after any re-indexing.
        """
        all_chunks = self._store.all_chunks()
        self._bm25_chunks = all_chunks
        tokenised = [_tokenise(c.content) for c in all_chunks]
        self._bm25 = BM25Okapi(tokenised)
        return self

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        user: Optional[UserContext] = None,
        filters: Optional[List[MetadataFilter]] = None,
    ) -> List[RetrievalResult]:
        """
        Hybrid search: dense + BM25 results merged via RRF.

        Falls back to dense-only if BM25 index has not been built.
        """
        query_vec = self._embedder.embed_query(query)

        # Dense retrieval — over-fetch before filtering
        dense = self._store.search(
            query_vec, k=k * 3, min_score=0.0,
            user=user, filters=filters,
        )

        if self._bm25 is None:
            # BM25 not built — dense only
            return dense[:k]

        # BM25 retrieval
        tokens = _tokenise(query)
        bm25_scores = self._bm25.get_scores(tokens)

        # Apply ACL + metadata filter to BM25 results
        sparse: List[RetrievalResult] = []
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:k * 3]
        for idx in top_bm25_idx:
            chunk = self._bm25_chunks[idx]
            score = float(bm25_scores[idx])
            if score <= 0:
                break
            # ACL check
            if user is not None:
                from src.acl import can_access
                if not can_access(user, chunk.metadata.get("permissions", [])):
                    continue
            # Metadata filter check
            if filters and not all(f.matches(chunk.metadata) for f in filters):
                continue
            sparse.append(RetrievalResult(chunk=chunk, score=score))

        # Merge with RRF
        merged = _rrf_merge([dense, sparse])

        # Apply min_score to RRF scores (RRF scores are small by design ~0.01-0.05)
        return merged[:k]
