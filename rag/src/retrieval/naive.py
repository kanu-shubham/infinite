"""
Naive RAG — the simplest possible retrieval-augmented generation pipeline.

How it works
------------
1. Embed the user query → dense vector q.
2. Search the vector store for the top-k most similar chunks.
3. Concatenate those chunks as context.
4. Send (context, query) to Claude and return the grounded answer.

       query
         │
    [Embedder]
         │ q_vec
    [VectorStore.search] ──→ top-k chunks
         │
    [Generator] ──→ answer + citations

Strengths
---------
- Dead simple to understand and debug.
- Very fast (single round-trip to the LLM).

Weaknesses
----------
- Query vocabulary mismatch: if the user asks about "ML" but documents
  say "machine learning", TF-IDF misses it (HyDE and query expansion fix this).
- Single shot: no reflection on retrieval quality (Self-RAG fixes this).
- Flat retrieval: misses cross-document reasoning (multi-hop fixes this).
- Long queries may dilute the embedding (query decomposition fixes this).
"""

from __future__ import annotations

from src.config import config
from src.generation.generator import RAGResponse
from .base import BaseRAG


class NaiveRAG(BaseRAG):
    """
    Embed → Retrieve → Generate.

    This is your baseline.  Every advanced pattern should outperform it
    on complex or ambiguous queries while matching it on simple ones.
    """

    def run(self, query: str) -> RAGResponse:
        # Step 1: retrieve
        results = self._retrieve(query, k=config.top_k)

        # Step 2: generate (handles no-result fallback internally)
        return self.generator.generate(
            query=query,
            results=results,
            pattern="naive",
        )
