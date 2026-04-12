"""
hybrid_retriever.py
===================
Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion (RRF) and cross-encoder reranking.

THEORY RECAP (see THEORY.md Sections 5 and 6):
  Why hybrid?
    BM25 strengths: exact keywords, rare terms, product names, codes
    Dense strengths: semantic similarity, paraphrasing, synonyms

  Fusion: combine ranked lists without worrying about score scale mismatch.
    BM25 scores are in range [0, ~20]
    Dense scores are in range [-1, 1]
    → Can't add them directly! Use rank-based fusion.

  RRF (Reciprocal Rank Fusion):
    score(d) = Σ 1/(k + rank(d, list_i))  where k=60

  Reranking: after fusion, apply a cross-encoder that jointly processes
    (query, document) for much more accurate relevance scoring.
"""

from typing import List, Dict, Optional, Tuple
from retrieval.bm25_retriever import BM25Retriever, SearchResult
from retrieval.dense_retriever import DenseRetriever


class HybridRetriever:
    """
    Combines BM25 and dense retrieval with RRF fusion and optional reranking.

    FULL PIPELINE:
        query
          ├─ BM25 search → top-50 (keyword matching)
          ├─ Dense search → top-50 (semantic matching)
          ↓
        RRF Fusion → top-20 (rank-based combination)
          ↓
        Cross-encoder Reranker → top-5 (precise scoring)
          ↓
        Final results → LLM context

    ARCHITECTURE DECISION:
        We fetch more candidates (50) than we need (5) at each stage.
        Each stage narrows down with increasing quality but decreasing speed:
          Stage 1 (BM25):    O(|Q|) — microseconds
          Stage 2 (Dense):   O(log N) — milliseconds (HNSW)
          Stage 3 (Rerank):  O(candidates × model_size) — 50-200ms
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        reranker=None,  # CrossEncoderReranker instance (optional)
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        fusion_top_k: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """
        Args:
            bm25_retriever:  Indexed BM25 retriever
            dense_retriever: Indexed dense retriever
            reranker:        Optional cross-encoder reranker
            bm25_top_k:      Candidates from BM25 (pre-fusion)
            dense_top_k:     Candidates from Dense (pre-fusion)
            fusion_top_k:    Candidates after RRF (pre-rerank)
            final_top_k:     Final results returned to caller
            rrf_k:           RRF constant (60 is standard, higher = gentler fusion)
            dense_weight:    Weight for dense scores in RRF (default 1.0 = equal)
            bm25_weight:     Weight for BM25 scores in RRF (default 1.0 = equal)
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.fusion_top_k = fusion_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
    ) -> List[Tuple[str, float]]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.

        ALGORITHM:
            For each document d:
              rrf_score(d) = dense_weight × 1/(k + rank_dense(d))
                           + bm25_weight × 1/(k + rank_bm25(d))

            If document only appears in one list:
              Only that list's term contributes.

        WHY k=60?
            Smaller k makes top ranks matter MORE (more aggressive).
            k=60 is the empirically validated default from the original paper.
            For RAG, k=60 works well.

        Returns:
            List of (chunk_id, rrf_score) sorted by score descending.
        """
        # Map chunk_id → rank for each retriever
        bm25_ranks: Dict[str, int] = {r.chunk_id: r.rank for r in bm25_results}
        dense_ranks: Dict[str, int] = {r.chunk_id: r.rank for r in dense_results}

        # Also need chunk metadata for results
        all_results: Dict[str, SearchResult] = {}
        for r in bm25_results:
            all_results[r.chunk_id] = r
        for r in dense_results:
            all_results[r.chunk_id] = r

        # Union of all chunk IDs
        all_chunk_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        # Compute RRF score for each
        rrf_scores = []
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                score += self.bm25_weight / (self.rrf_k + bm25_ranks[chunk_id])
            rrf_scores.append((chunk_id, score, all_results[chunk_id]))

        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores

    def search(
        self,
        query: str,
        verbose: bool = False,
    ) -> List[SearchResult]:
        """
        Full hybrid search: BM25 + Dense → RRF → Reranker → top results.

        Args:
            query:    user query string
            verbose:  print intermediate results for learning/debugging

        Returns:
            Top-k SearchResult objects, sorted by final relevance.
        """
        # ── Stage 1a: BM25 Search ──────────────────────────────────────────
        bm25_results = self.bm25.search(query, top_k=self.bm25_top_k)

        # ── Stage 1b: Dense Search ─────────────────────────────────────────
        dense_results = self.dense.search(query, top_k=self.dense_top_k)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: '{query}'")
            print(f"{'='*60}")
            print(f"\nBM25 top-3 (keyword matching):")
            for r in bm25_results[:3]:
                print(f"  [{r.rank}] score={r.score:.3f} '{r.text[:80]}...'")
            print(f"\nDense top-3 (semantic matching):")
            for r in dense_results[:3]:
                print(f"  [{r.rank}] score={r.score:.4f} '{r.text[:80]}...'")

        # ── Stage 2: RRF Fusion ────────────────────────────────────────────
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)
        top_fused = fused[:self.fusion_top_k]

        # Rebuild SearchResult objects from fused list
        fused_results = []
        for rank, (chunk_id, rrf_score, original_result) in enumerate(top_fused):
            fused_results.append(SearchResult(
                chunk_id=original_result.chunk_id,
                doc_id=original_result.doc_id,
                text=original_result.text,
                score=rrf_score,
                rank=rank + 1,
                metadata=original_result.metadata,
            ))

        if verbose:
            print(f"\nAfter RRF fusion (top-5 of {len(fused_results)}):")
            for r in fused_results[:5]:
                in_bm25 = r.chunk_id in {x.chunk_id for x in bm25_results}
                in_dense = r.chunk_id in {x.chunk_id for x in dense_results}
                sources = []
                if in_bm25: sources.append("BM25")
                if in_dense: sources.append("Dense")
                print(f"  [{r.rank}] rrf={r.score:.5f} [{'+'.join(sources)}] '{r.text[:80]}...'")

        # ── Stage 3: Reranking (optional but recommended) ──────────────────
        if self.reranker is not None:
            final_results = self.reranker.rerank(
                query=query,
                results=fused_results,
                top_k=self.final_top_k,
            )
            if verbose:
                print(f"\nAfter cross-encoder reranking (top-{self.final_top_k}):")
                for r in final_results:
                    print(f"  [{r.rank}] score={r.score:.4f} '{r.text[:80]}...'")
        else:
            final_results = fused_results[:self.final_top_k]
            # Re-rank numbers
            for i, r in enumerate(final_results):
                r.rank = i + 1

        return final_results

    def search_diagnostic(self, query: str) -> dict:
        """
        Returns detailed diagnostic info showing contribution of each retriever.
        Useful for understanding how hybrid search works.
        """
        bm25_results = self.bm25.search(query, top_k=self.bm25_top_k)
        dense_results = self.dense.search(query, top_k=self.dense_top_k)
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        bm25_ids = {r.chunk_id for r in bm25_results}
        dense_ids = {r.chunk_id for r in dense_results}

        diagnostics = []
        for rank, (chunk_id, rrf_score, result) in enumerate(fused[:10]):
            diagnostics.append({
                "rank": rank + 1,
                "chunk_id": chunk_id,
                "rrf_score": round(rrf_score, 6),
                "in_bm25": chunk_id in bm25_ids,
                "in_dense": chunk_id in dense_ids,
                "bm25_rank": next((r.rank for r in bm25_results if r.chunk_id == chunk_id), None),
                "dense_rank": next((r.rank for r in dense_results if r.chunk_id == chunk_id), None),
                "text_preview": result.text[:100],
            })

        return {
            "query": query,
            "bm25_retrieved": len(bm25_results),
            "dense_retrieved": len(dense_results),
            "union_size": len(set(bm25_ids) | set(dense_ids)),
            "top_10": diagnostics,
        }


class CrossEncoderReranker:
    """
    Cross-encoder reranker: scores (query, document) pairs jointly.

    THEORY RECAP (see THEORY.md Section 6):
        Cross-encoders process [CLS] query [SEP] document [SEP] as a single input.
        Full self-attention lets every query token attend to every document token.
        This is far more powerful than comparing separate embeddings (bi-encoder).

        Tradeoff:
          - Slower: can't pre-compute document vectors
          - More accurate: sees both query and document at once
          - Only feasible for small candidate sets (typically 20-50)

    MODEL NOTE:
        We use cross-encoder/ms-marco-MiniLM-L-6-v2:
          - Trained on MS MARCO passage ranking (522K queries, 8.8M passages)
          - 22MB — very fast even on CPU
          - Good general-purpose reranker
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: HuggingFace cross-encoder model
        """
        self.model_name = model_name
        print(f"Loading cross-encoder reranker: {model_name}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        print("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Score each (query, document) pair and return top-k.

        HOW IT WORKS:
            For each result, create input: [CLS] query [SEP] document [SEP]
            Pass all pairs through the cross-encoder in a batch.
            Get a single relevance score per pair.
            Sort by score, return top-k.

        Args:
            query:   user query
            results: candidate results to rerank
            top_k:   number of final results to return

        Returns:
            Top-k results sorted by cross-encoder score.
        """
        if not results:
            return []

        # Create (query, passage) pairs for cross-encoder
        # Cross-encoder expects: [[query1, doc1], [query1, doc2], ...]
        pairs = [[query, r.text] for r in results]

        # Score all pairs (batched)
        scores = self.model.predict(pairs)

        # Create new results with cross-encoder scores
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                score=float(score),
                rank=0,  # will be set below
                metadata=result.metadata,
            ))

        # Sort by cross-encoder score (descending) and assign ranks
        reranked.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(reranked[:top_k]):
            r.rank = i + 1

        return reranked[:top_k]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/infinite/rag")
    from corpus.corpus_loader import load_ai_corpus
    from corpus.chunkers import RecursiveCharacterChunker
    from embeddings.embedder import DenseEmbedder

    # Build the full system
    corpus = load_ai_corpus()
    chunker = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_corpus(corpus)

    # Index both retrievers
    bm25 = BM25Retriever()
    bm25.index(chunks)

    embedder = DenseEmbedder()
    dense = DenseRetriever(embedder)
    dense.index(chunks)

    # Create hybrid retriever (no reranker for this quick test)
    hybrid = HybridRetriever(bm25, dense, final_top_k=5)

    # Test
    query = "How does self-attention work in transformer models?"
    print(f"\n{'='*60}")
    print(f"Hybrid Search Diagnostic")
    print(f"{'='*60}")
    diag = hybrid.search_diagnostic(query)
    print(f"Query: '{diag['query']}'")
    print(f"BM25 retrieved: {diag['bm25_retrieved']}")
    print(f"Dense retrieved: {diag['dense_retrieved']}")
    print(f"Union: {diag['union_size']}")
    print(f"\nTop-5 after RRF:")
    for d in diag['top_10'][:5]:
        sources = []
        if d['in_bm25']: sources.append(f"BM25@{d['bm25_rank']}")
        if d['in_dense']: sources.append(f"Dense@{d['dense_rank']}")
        print(f"  [{d['rank']}] rrf={d['rrf_score']:.5f} [{', '.join(sources)}]")
        print(f"       '{d['text_preview']}...'")
