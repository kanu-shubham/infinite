"""
hybrid.py — Combine BM25 keyword results + vector semantic results into one ranked list.

Problem:
    BM25 is great for exact keywords ("ISO 27001") but misses paraphrases.
    Vectors catch paraphrases ("information security standard") but miss rare keywords.
    Hybrid search gets both.

Solution: Reciprocal Rank Fusion (RRF).
    Instead of trying to normalise BM25 scores (unbounded) and cosine scores (0–1)
    onto the same scale — which is mathematically tricky — we use only the RANKS.

    RRF formula:   score(doc) = Σ  1 / (k + rank_in_list)
                               each list

    k = 60 is a constant that prevents the #1 result from dominating too much.
    A doc ranked 1st in both lists gets: 1/(60+1) + 1/(60+1) ≈ 0.033
    A doc ranked 1st in only one list gets:  1/(60+1) ≈ 0.016
"""
from documents import Chunk
from vectorstore import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[Chunk]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Merge multiple ranked lists of Chunks using RRF.

    Args:
        result_lists: e.g. [[bm25_result1, bm25_result2, ...],
                             [vector_result1, vector_result2, ...]]
        k: smoothing constant (60 is the standard default from the 2009 paper)

    Returns:
        List of (chunk, rrf_score) sorted by rrf_score descending.
    """
    # rrf_scores maps chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # best_chunk maps chunk_id → the actual Chunk object
    best_chunk: dict[str, Chunk] = {}

    for ranked_list in result_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # rank starts at 1 (not 0)
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            best_chunk[cid] = chunk

    # Sort by accumulated RRF score, highest first
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(best_chunk[cid], score) for cid, score in merged]


class HybridRetriever:
    """
    Runs BM25 and vector search in parallel, merges with RRF.
    """

    def __init__(self, bm25_index, vector_store, embedder):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 10, bm25_k: int = 20, vec_k: int = 20) -> list[RetrievalResult]:
        # ── BM25 leg ──────────────────────────────────────────────────────
        bm25_results = self.bm25_index.search(query, top_k=bm25_k)
        bm25_chunks = [chunk for chunk, _score in bm25_results]

        # ── Vector leg ────────────────────────────────────────────────────
        q_emb = self.embedder.embed_query(query)
        vec_results = self.vector_store.search(q_emb, top_k=vec_k)
        vec_chunks = [r.chunk for r in vec_results]

        # ── Merge with RRF ────────────────────────────────────────────────
        merged = reciprocal_rank_fusion([bm25_chunks, vec_chunks], k=60)

        # Re-wrap as RetrievalResult for a uniform return type
        return [
            RetrievalResult(chunk=chunk, score=score)
            for chunk, score in merged[:top_k]
        ]
