"""
Query Decomposition — RAG-Fusion / Least-to-Most Decomposition
(inspired by Khattab et al. DSP, LangChain RAG-Fusion)

Core Idea
---------
Complex or compound queries dilute the query embedding, causing poor
retrieval.  Query decomposition:

  1. Breaks a complex question into N simpler, focused sub-queries.
  2. Retrieves documents independently for each sub-query.
  3. Merges all retrieved documents using Reciprocal Rank Fusion (RRF).
  4. Generates a single coherent answer from the merged context.

Example
-------
  Original: "Compare the training data and model architecture of GPT-4 and LLaMA-2"

  Sub-queries:
    1. "What training data was used for GPT-4?"
    2. "What is the model architecture of GPT-4?"
    3. "What training data was used for LLaMA-2?"
    4. "What is the model architecture of LLaMA-2?"

Reciprocal Rank Fusion (RRF)
-----------------------------
A classic information-retrieval technique for merging ranked lists:

  RRF_score(d) = Σ_{q} 1 / (k + rank_q(d))

where k=60 is a smoothing constant.  Documents that appear high in
multiple sub-query result lists score highest.

When does query decomposition help?
------------------------------------
- Compound questions with AND / OR structure.
- Comparative questions ("Compare A vs B on dimension X").
- Multi-aspect questions ("Describe X in terms of Y, Z, and W").

Limitation: Sub-queries that are too generic may retrieve overlapping
documents, wasting context window space.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import List, Tuple

import anthropic

from src.config import config
from src.documents import RetrievalResult
from src.generation.generator import RAGResponse
from .base import BaseRAG


_DECOMPOSE_SYSTEM = textwrap.dedent("""\
    You are an expert at breaking down complex research questions into
    precise, non-overlapping sub-questions.

    Given a question, produce a numbered list of 2–5 focused sub-questions
    that together cover all aspects of the original question.

    Rules:
    - Each sub-question should be answerable independently.
    - Sub-questions should be concrete and search-friendly.
    - Avoid sub-questions that are too broad or too similar to each other.
    - Output ONLY the numbered list, no preamble or explanation.

    Example:
    Question: "How does BERT differ from GPT in architecture and training?"
    1. What is the architecture of BERT?
    2. What training objective does BERT use?
    3. What is the architecture of GPT?
    4. What training objective does GPT use?
""")


def _reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """
    Merge multiple ranked retrieval lists using Reciprocal Rank Fusion.

    Returns a single list of RetrievalResult sorted by fused score,
    with the best chunk object kept for each unique chunk_id.
    """
    scores: dict[str, float] = defaultdict(float)
    best_chunks: dict[str, "Chunk"] = {}
    best_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk.chunk_id
            scores[cid] += 1.0 / (k + rank)
            # Keep the highest-scored Chunk object for each unique id
            if cid not in best_scores or result.score > best_scores[cid]:
                best_chunks[cid] = result.chunk
                best_scores[cid] = result.score

    fused = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [
        RetrievalResult(chunk=best_chunks[cid], score=scores[cid])
        for cid in fused
    ]


class QueryDecompRAG(BaseRAG):
    """
    Decompose → Retrieve independently → RRF merge → Generate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _decompose(self, query: str) -> List[str]:
        """Break *query* into focused sub-queries."""
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=256,
            temperature=0,
            system=_DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": f"Question: {query}"}],
        )
        lines = resp.content[0].text.strip().splitlines()
        sub_queries = []
        for line in lines:
            # Strip list markers: "1. ", "- ", etc.
            stripped = line.strip().lstrip("0123456789.-) ").strip()
            if stripped:
                sub_queries.append(stripped)
        # Fallback: if decomposition fails, just use the original
        return sub_queries or [query]

    def run(self, query: str) -> RAGResponse:
        # Step 1: Decompose
        sub_queries = self._decompose(query)

        # Step 2: Retrieve independently for each sub-query
        ranked_lists: List[List[RetrievalResult]] = []
        for sq in sub_queries:
            results = self._retrieve(sq, k=config.top_k)
            if results:
                ranked_lists.append(results)

        # Step 3: Merge with RRF
        if ranked_lists:
            merged = _reciprocal_rank_fusion(ranked_lists)[:config.top_k]
        else:
            # Fallback to naive retrieval if decomposition yields nothing
            merged = self._retrieve(query, k=config.top_k)

        # Step 4: Generate
        response = self.generator.generate(
            query=query,
            results=merged,
            pattern="query_decomp",
        )
        response.metadata["sub_queries"] = sub_queries
        response.metadata["n_sub_queries"] = len(sub_queries)
        return response
