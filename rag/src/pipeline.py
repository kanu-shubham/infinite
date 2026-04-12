"""
RAG Pipeline Orchestrator

This is the entry point for building and querying a RAG system.
It wires together all components and exposes a clean API:

  pipeline = RAGPipeline()
  pipeline.index(documents)
  result = pipeline.query("What is HyDE?", pattern="hyde")

Architecture Overview
---------------------

  Documents → [Chunker] → Chunks
                              │
                         [Embedder.fit()]   ← must see full corpus
                              │
                    [VectorStore.add_chunks()]
                              │
         ┌────────────────────┼──────────────────────┐
         │                    │                       │
    [NaiveRAG]          [HyDERAG]            [SelfRAG] …
         │                    │                       │
         └────────────────────┴───────────────────────┘
                              │
                        [Generator]
                              │
                        RAGResponse
                              │
              ┌───────────────┴────────────────┐
         [Faithfulness]               [Relevance]
           Evaluator                  Evaluator

Edge Cases Handled
------------------
1. No results     — scores below min_similarity → fallback answer
2. Conflicting    — Generator detects and surfaces conflicting sources
3. Empty corpus   — index() with no docs raises a clear error
4. Bad query      — empty string → raises ValueError
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import config
from src.documents import Document, Chunk, chunk_document
from src.embeddings.tfidf import TFIDFEmbedder
from src.generation.generator import Generator, RAGResponse
from src.metrics.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from src.metrics.relevance import RelevanceEvaluator, RelevanceResult
from src.retrieval.naive import NaiveRAG
from src.retrieval.hyde import HyDERAG
from src.retrieval.self_rag import SelfRAG
from src.retrieval.raptor import RaptorRAG
from src.retrieval.multi_hop import MultiHopRAG
from src.retrieval.query_decomp import QueryDecompRAG
from src.vectorstore.memory import InMemoryVectorStore


PATTERNS = {
    "naive": NaiveRAG,
    "hyde": HyDERAG,
    "self_rag": SelfRAG,
    "raptor": RaptorRAG,
    "multi_hop": MultiHopRAG,
    "query_decomp": QueryDecompRAG,
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
    High-level RAG pipeline that manages the full lifecycle.

    Parameters
    ----------
    chunk_size    : characters per chunk (default from config)
    chunk_overlap : character overlap between consecutive chunks
    n_components  : LSA embedding dimensions (more = richer, slower)
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        n_components: int = 128,
    ):
        config.validate()

        self._chunk_size = chunk_size or config.chunk_size
        self._chunk_overlap = chunk_overlap or config.chunk_overlap

        self._embedder = TFIDFEmbedder(n_components=n_components)
        self._store = InMemoryVectorStore()
        self._generator = Generator()
        self._indexed = False

        # Lazy-init per pattern
        self._strategies: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: List[Document]) -> "RAGPipeline":
        """
        Chunk and index all documents.

        This must be called before any query.  Calling index() again
        clears the store and re-indexes from scratch.

        Returns self for chaining.
        """
        if not documents:
            raise ValueError("document list is empty — nothing to index")

        self._store.clear()
        self._strategies.clear()

        # 1. Chunk
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = chunk_document(doc, self._chunk_size, self._chunk_overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks produced — documents may be empty")

        # 2. Fit embedder on the full corpus (critical for TF-IDF IDF weights)
        texts = [c.content for c in all_chunks]
        self._embedder.fit(texts)

        # 3. Embed + store
        embeddings = self._embedder.embed_texts(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        self._store.add_chunks(all_chunks)
        self._indexed = True

        print(f"Indexed {len(documents)} documents → {len(all_chunks)} chunks")
        return self

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        pattern: str = "naive",
        evaluate: bool = False,
    ) -> RAGResponse | EvaluationResult:
        """
        Run a query against the indexed corpus.

        Parameters
        ----------
        question : the user's question (non-empty string)
        pattern  : one of "naive", "hyde", "self_rag", "raptor",
                           "multi_hop", "query_decomp"
        evaluate : if True, also compute faithfulness + relevance metrics
                   (adds ~3-5 extra LLM calls)

        Returns
        -------
        RAGResponse if evaluate=False
        EvaluationResult if evaluate=True
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        if not self._indexed:
            raise RuntimeError("Call pipeline.index(documents) before querying")

        if pattern not in PATTERNS:
            raise ValueError(
                f"Unknown pattern '{pattern}'. "
                f"Valid options: {list(PATTERNS.keys())}"
            )

        strategy = self._get_strategy(pattern)
        response = strategy.run(question)

        if not evaluate:
            return response

        # Optional metric evaluation
        f_eval = FaithfulnessEvaluator()
        r_eval = RelevanceEvaluator()

        faithfulness = f_eval.evaluate(response)
        relevance = r_eval.evaluate(response)

        return EvaluationResult(
            response=response,
            faithfulness=faithfulness,
            relevance=relevance,
        )

    def compare(
        self,
        question: str,
        patterns: Optional[List[str]] = None,
    ) -> Dict[str, RAGResponse]:
        """
        Run the same question through multiple patterns and return all results.

        Useful for side-by-side comparison during development.
        """
        patterns = patterns or list(PATTERNS.keys())
        return {p: self.query(question, pattern=p) for p in patterns}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_strategy(self, pattern: str):
        """Instantiate strategy objects lazily and cache them."""
        if pattern not in self._strategies:
            cls = PATTERNS[pattern]
            strategy = cls(
                store=self._store,
                embedder=self._embedder,
                generator=self._generator,
            )
            # RAPTOR needs extra setup
            if pattern == "raptor":
                print("Building RAPTOR tree (one-time, may take a moment)…")
                strategy.build_tree()
            self._strategies[pattern] = strategy
        return self._strategies[pattern]

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    @property
    def is_indexed(self) -> bool:
        return self._indexed
