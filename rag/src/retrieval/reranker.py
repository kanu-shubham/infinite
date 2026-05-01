"""
LLM-based Reranker

The two-stage retrieval pattern used in production:

  Stage 1 — Recall (cheap, fast):
    Retrieve top-20 candidates using vector similarity.
    Optimise for recall — get all the right answers in the set.

  Stage 2 — Precision (expensive, accurate):
    Reranker sees (query, chunk) together and scores relevance.
    Reranks the 20 candidates → returns top-5.
    Optimise for precision — right answers at the top.

Why reranking improves quality
-------------------------------
Vector similarity compares query and chunk embeddings independently.
It cannot model interactions like:
  - "The query asks about X but this passage is about a different X"
  - "This passage is relevant but buries the answer in the last sentence"

A reranker (cross-encoder) sees query and passage TOGETHER and can
model these interactions. In benchmarks, a good reranker on top of
vector retrieval outperforms vector retrieval alone by 10-20% on NDCG.

Two reranker options
---------------------
1. Cross-encoder model (e.g. BAAI/bge-reranker-base):
   - Runs locally, fast, no API cost
   - Requires torch: pip install sentence-transformers torch

2. LLM-as-reranker (implemented here):
   - Uses Claude to score each (query, chunk) pair
   - No extra dependencies
   - Slower and more expensive than a cross-encoder
   - Better for complex or domain-specific queries

For production: use a cross-encoder model for speed, LLM reranker
as a fallback or for high-value queries.
"""

from __future__ import annotations

import textwrap
from typing import List

import anthropic

from src.config import config
from src.documents import RetrievalResult


_SCORE_SYSTEM = textwrap.dedent("""\
    You are a relevance judge for a search engine.

    Given a search query and a passage, rate how relevant the passage
    is for answering the query on a scale of 0–3:

      0 = not relevant at all
      1 = tangentially related but doesn't answer the query
      2 = partially relevant, contains some useful information
      3 = highly relevant, directly answers or strongly supports the query

    Respond with ONLY the integer (0, 1, 2, or 3). No explanation.
""")


class LLMReranker:
    """
    Reranks retrieval results using Claude as a cross-encoder judge.

    Parameters
    ----------
    top_n : only consider top_n candidates from the initial retrieval
    """

    def __init__(self, top_n: int = 20):
        self._top_n = top_n
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Rerank *results* for *query*, returning top *k*.

        Each candidate is scored independently.  The scores are used
        to re-sort the list; the original similarity scores are preserved
        in result.score for reference.

        Parameters
        ----------
        query   : the user's question
        results : initial retrieval results (will use top self.top_n)
        k       : number of results to return after reranking
        """
        if not results:
            return []

        candidates = results[:self._top_n]

        # Score each candidate
        scored: List[tuple] = []
        for r in candidates:
            score = self._score(query, r.chunk.content)
            scored.append((score, r))

        # Sort by reranker score descending, break ties by original sim score
        scored.sort(key=lambda x: (x[0], x[1].score), reverse=True)

        return [r for _, r in scored[:k]]

    def _score(self, query: str, passage: str) -> int:
        """Score a single (query, passage) pair. Returns 0–3."""
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=4,
            temperature=0,
            system=_SCORE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Query: {query}\n\nPassage: {passage[:600]}",
            }],
        )
        text = resp.content[0].text.strip()
        try:
            return int(text[0])
        except (ValueError, IndexError):
            return 1  # neutral fallback
