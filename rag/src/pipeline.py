"""
RAG Pipeline Orchestrator — Production Version

Wires together every component and exposes a clean API:

    pipeline = RAGPipeline()
    pipeline.index(documents)
    result = pipeline.query("What is HyDE?", pattern="hyde")

New in this version vs the learning version
-------------------------------------------
  ✓ ACL — permission pre-filtering per user
  ✓ Metadata filtering — filter by date, author, source, etc.
  ✓ FAISS vector store — persistent, scales to millions of chunks
  ✓ Incremental indexing — upsert/delete individual documents
  ✓ Hybrid search — BM25 + dense combined via RRF
  ✓ Conversation history — query rewriter for multi-turn sessions
  ✓ Reranker — optional two-stage retrieval for higher precision
  ✓ Query result cache — identical queries served instantly
  ✓ Structured logging — every event logged as JSON
  ✓ Retry logic — exponential backoff on API errors

Architecture
------------

  Documents → [IncrementalIndexManager] → [FAISSVectorStore]
                                                  │
              ┌───────────────────────────────────┤
              │                                   │
       [ConversationRewriter]              [HybridRetriever]
              │                                   │
              └──────────────────┬────────────────┘
                                 │
                          [RAG Pattern]  (naive/hyde/self_rag/...)
                                 │
                          [LLMReranker]  (optional)
                                 │
                          [Generator]   (with retry + caching)
                                 │
                           RAGResponse
                                 │
              ┌──────────────────┴────────────────┐
         [Faithfulness]                    [Relevance]
           Evaluator                       Evaluator
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.acl import UserContext, SUPERUSER
from src.cache import QueryCache
from src.config import config
from src.documents import Document, Chunk
from src.embeddings.tfidf import TFIDFEmbedder
from src.generation.generator import Generator, RAGResponse
from src.indexing.incremental import IncrementalIndexManager
from src.metrics.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from src.metrics.relevance import RelevanceEvaluator, RelevanceResult
from src.observability import logger
from src.retrieval.conversation import ConversationRewriter
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.naive import NaiveRAG
from src.retrieval.hyde import HyDERAG
from src.retrieval.self_rag import SelfRAG
from src.retrieval.raptor import RaptorRAG
from src.retrieval.multi_hop import MultiHopRAG
from src.retrieval.query_decomp import QueryDecompRAG
from src.retrieval.graph_rag import GraphRAG
from src.retrieval.reranker import LLMReranker
from src.vectorstore.memory import InMemoryVectorStore
from src.vectorstore.faiss_store import FAISSVectorStore, MetadataFilter


PATTERNS = {
    "naive":        NaiveRAG,
    "hyde":         HyDERAG,
    "self_rag":     SelfRAG,
    "raptor":       RaptorRAG,
    "multi_hop":    MultiHopRAG,
    "query_decomp": QueryDecompRAG,
    "graph_rag":    GraphRAG,
}


@dataclass
class EvaluationResult:
    response: RAGResponse
    faithfulness: Optional[FaithfulnessResult] = None
    relevance: Optional[RelevanceResult] = None

    def report(self) -> str:
        lines = [self.response.pretty(), ""]
        lines.append("── Metrics ─────────────────────────────────────")
        if self.faithfulness:
            lines.append(f"  {self.faithfulness}")
        else:
            lines.append("  Faithfulness: (not evaluated)")
        if self.relevance:
            lines.append(f"  {self.relevance}")
        else:
            lines.append("  Relevance: (not evaluated)")
        lines.append("─────────────────────────────────────────────────")
        return "\n".join(lines)


class RAGPipeline:
    """
    Production RAG pipeline.

    Parameters
    ----------
    use_faiss       : use FAISS persistent store instead of in-memory NumPy
    use_hybrid      : combine BM25 sparse + dense retrieval
    enable_reranker : run LLM reranker as second retrieval stage
    cache_ttl       : seconds to cache query results (0 = disabled)
    chunk_size      : characters per chunk
    chunk_overlap   : character overlap between consecutive chunks
    n_components    : LSA embedding dimensions (TF-IDF only)
    index_path      : directory for FAISS persistence (use_faiss only)
    """

    def __init__(
        self,
        use_faiss: bool = False,
        use_hybrid: bool = False,
        enable_reranker: bool = False,
        cache_ttl: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        n_components: int = 128,
        index_path: str = None,
    ):
        config.validate()

        self._chunk_size = chunk_size or config.chunk_size
        self._chunk_overlap = chunk_overlap or config.chunk_overlap

        # --- Embedder ---
        self._embedder = TFIDFEmbedder(n_components=n_components)

        # --- Vector store ---
        if use_faiss:
            ipath = index_path or config.index_path
            self._store = FAISSVectorStore(dim=n_components, index_path=ipath)
            self._store.load()   # no-op if first run
        else:
            self._store = InMemoryVectorStore()

        # --- Incremental index manager ---
        self._index_manager = IncrementalIndexManager(
            store=self._store,
            embedder=self._embedder,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        # --- Generator ---
        self._generator = Generator()

        # --- Optional components ---
        self._hybrid: Optional[HybridRetriever] = None
        self._use_hybrid = use_hybrid

        self._reranker: Optional[LLMReranker] = None
        if enable_reranker:
            self._reranker = LLMReranker(top_n=config.reranker_top_n)

        self._rewriter = ConversationRewriter()

        ttl = cache_ttl if cache_ttl is not None else config.cache_ttl
        self._cache = QueryCache(ttl_seconds=ttl) if ttl > 0 else None

        # Lazy-init per pattern
        self._strategies: Dict[str, object] = {}
        self._indexed = False

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[Document]) -> "RAGPipeline":
        """
        Full index build — chunks, embeds, and stores all documents.

        For large corpora after initial indexing, prefer upsert_document()
        to update individual documents without rebuilding everything.
        """
        if not documents:
            raise ValueError("document list is empty — nothing to index")

        self._store.clear()
        self._strategies.clear()
        if self._cache:
            self._cache.invalidate()

        # Fit embedder on full corpus first (critical for TF-IDF IDF weights)
        from src.documents import chunk_document
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_document(doc, self._chunk_size, self._chunk_overlap))

        if not all_chunks:
            raise ValueError("No chunks produced — documents may be empty")

        texts = [c.content for c in all_chunks]
        self._embedder.fit(texts)

        # Now use incremental manager to embed + store
        self._index_manager._embedder = self._embedder
        self._index_manager.batch_upsert(documents)

        # Build hybrid BM25 index if requested
        if self._use_hybrid:
            self._hybrid = HybridRetriever(self._store, self._embedder)
            self._hybrid.build_bm25()

        # Save FAISS index to disk if using FAISS
        if isinstance(self._store, FAISSVectorStore):
            self._store.save()

        self._indexed = True
        print(f"Indexed {len(documents)} documents → {self.chunk_count} chunks")
        return self

    def upsert_document(self, doc: Document) -> None:
        """
        Add or update a single document without rebuilding the full index.

        Use this for incremental updates (webhook-driven, scheduled sync).
        """
        self._index_manager.upsert(doc)
        if self._use_hybrid and self._hybrid:
            self._hybrid.build_bm25()    # rebuild BM25 over updated corpus
        if isinstance(self._store, FAISSVectorStore):
            self._store.save()
        if self._cache:
            self._cache.invalidate()     # stale responses may no longer be valid

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and all its chunks from the index."""
        self._index_manager.delete(doc_id)
        if self._use_hybrid and self._hybrid:
            self._hybrid.build_bm25()
        if isinstance(self._store, FAISSVectorStore):
            self._store.save()
        if self._cache:
            self._cache.invalidate()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        pattern: str = "naive",
        evaluate: bool = False,
        user: Optional[UserContext] = None,
        filters: Optional[List[MetadataFilter]] = None,
        history: Optional[List[dict]] = None,
    ) -> "RAGResponse | EvaluationResult":
        """
        Run a query against the indexed corpus.

        Parameters
        ----------
        question : the user's question
        pattern  : "naive" | "hyde" | "self_rag" | "raptor" |
                   "multi_hop" | "query_decomp"
        evaluate : also compute faithfulness + relevance metrics
        user     : UserContext for ACL filtering (None = superuser)
        filters  : metadata filters (date, author, source, etc.)
        history  : conversation history for multi-turn rewriting
                   [{"role": "user", "content": "..."}, ...]
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not self._indexed:
            raise RuntimeError("Call pipeline.index(documents) before querying")
        if pattern not in PATTERNS:
            raise ValueError(f"Unknown pattern '{pattern}'. Valid: {list(PATTERNS.keys())}")

        effective_user = user or SUPERUSER
        t0 = time.time()
        logger.query_start(question, pattern, user_id=getattr(user, "user_id", None))

        # --- Step 1: Rewrite query if conversation history provided ---
        retrieval_query = question
        if history:
            retrieval_query = self._rewriter.rewrite(question, history)

        # --- Step 2: Cache check ---
        if self._cache:
            cached = self._cache.get(retrieval_query, pattern)
            if cached:
                logger.cache_hit(retrieval_query, pattern)
                return EvaluationResult(response=cached) if evaluate else cached

        # --- Step 3: Inject ACL + filters into the store ---
        self._patch_store_for_user(effective_user, filters)

        # --- Step 4: Run RAG pattern ---
        strategy = self._get_strategy(pattern)
        response = strategy.run(retrieval_query)

        # Restore original query in response (before rewriting)
        response.query = question

        # --- Step 5: Rerank (optional) ---
        if self._reranker and response.retrieval_results:
            reranked = self._reranker.rerank(
                question, response.retrieval_results, k=config.top_k
            )
            response.retrieval_results = reranked
            response.citations = response.citations[:len(reranked)]

        # --- Step 6: Log ---
        latency = (time.time() - t0) * 1000
        scores = [r.score for r in response.retrieval_results]
        logger.retrieval(question, len(scores), scores, latency, response.fallback)
        logger.generation(
            question, pattern, latency,
            response.metadata.get("input_tokens", 0),
            response.metadata.get("output_tokens", 0),
            conflict=response.conflict_detected,
        )

        # --- Step 7: Cache result ---
        if self._cache:
            self._cache.set(retrieval_query, pattern, response)

        if not evaluate:
            return response

        # --- Step 8: Evaluate (optional) ---
        faithfulness = FaithfulnessEvaluator().evaluate(response)
        relevance = RelevanceEvaluator().evaluate(response)
        logger.metrics(
            question,
            faithfulness.score,
            relevance.context_precision,
            relevance.context_recall,
            relevance.answer_relevance,
        )
        return EvaluationResult(response=response,
                                faithfulness=faithfulness,
                                relevance=relevance)

    def compare(
        self,
        question: str,
        patterns: Optional[List[str]] = None,
        user: Optional[UserContext] = None,
    ) -> Dict[str, RAGResponse]:
        """Run the same question through multiple patterns side-by-side."""
        patterns = patterns or list(PATTERNS.keys())
        return {p: self.query(question, pattern=p, user=user) for p in patterns}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _patch_store_for_user(
        self,
        user: UserContext,
        filters: Optional[List[MetadataFilter]],
    ) -> None:
        """
        Inject user + filters into the store so all retrieval strategies
        (naive, HyDE, multi-hop, etc.) automatically apply ACL without
        each strategy needing to know about it.

        We monkey-patch the store's search method temporarily for this request.
        A cleaner production approach is a middleware/decorator pattern.
        """
        original_search = self._store.search

        def _filtered_search(query_vec, k=5, min_score=0.0, **kwargs):
            if isinstance(self._store, FAISSVectorStore):
                return original_search(query_vec, k=k, min_score=min_score,
                                       user=user, filters=filters)
            # InMemoryVectorStore doesn't have ACL — apply post-hoc
            results = original_search(query_vec, k=k * 3, min_score=min_score)
            from src.acl import can_access
            permitted = [
                r for r in results
                if can_access(user, r.chunk.metadata.get("permissions", []))
                and (not filters or all(f.matches(r.chunk.metadata) for f in filters))
            ]
            return permitted[:k]

        self._store.search = _filtered_search

    def _get_strategy(self, pattern: str):
        if pattern not in self._strategies:
            cls = PATTERNS[pattern]
            strategy = cls(
                store=self._store,
                embedder=self._embedder,
                generator=self._generator,
            )
            if pattern == "raptor":
                print("Building RAPTOR tree (one-time)…")
                strategy.build_tree()
            self._strategies[pattern] = strategy
        return self._strategies[pattern]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    def cache_stats(self) -> dict:
        return self._cache.stats() if self._cache else {"enabled": False}

    def index_stats(self) -> dict:
        s = self._index_manager.stats()
        return {"docs": s.total_docs, "chunks": s.total_chunks}
