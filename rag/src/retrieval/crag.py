"""
CRAG — Corrective Retrieval Augmented Generation
(Yan et al., 2024 — arxiv.org/abs/2401.15884)

The Problem with Naive RAG
--------------------------
Standard RAG blindly trusts whatever comes back from retrieval.
If the top chunks are irrelevant, the LLM hallucinates anyway —
now with fabricated citations to boot.

CRAG adds a self-correction loop: after retrieval, it evaluates
the relevance of what was retrieved, and takes corrective action:

  ┌──────────────────────────────────────────────────────┐
  │  Relevance score  │  Action                          │
  ├──────────────────────────────────────────────────────┤
  │  HIGH (≥ 0.7)     │  Proceed — use retrieved chunks  │
  │  MEDIUM (0.3–0.7) │  Refine — strip irrelevant parts │
  │  LOW  (< 0.3)     │  Correct — rewrite query,        │
  │                   │  re-retrieve (+ web if available) │
  └──────────────────────────────────────────────────────┘

In the original paper, the LOW path triggers a web search.
This implementation falls back to a rewritten query against the
same vector store (web search optional via a pluggable hook).

CRAG vs Self-RAG
----------------
Self-RAG adds reflection TOKENS to the LLM's generation vocabulary
so the model itself decides when to retrieve and whether results
are useful.  This requires a fine-tuned model.

CRAG is a wrapper around ANY retriever — no fine-tuning needed.
It adds an evaluator LLM call, not a fine-tuned generator.

When to use CRAG
----------------
- When retrieval quality is uncertain (mixed-quality corpus)
- When you have noisy or heterogeneous document collections
- When hallucination from bad retrieval is a real risk
- As a safety net in production over Self-RAG
"""

from __future__ import annotations

import textwrap

import anthropic

from src.config import config
from src.generation.generator import RAGResponse
from src.retrieval.base import BaseRAG


_RELEVANCE_SYSTEM = textwrap.dedent("""\
    You are a retrieval quality judge.

    Given a query and a set of retrieved passages, assess how well the
    passages answer the query on a scale of 0.0 to 1.0:

      0.0 = completely irrelevant, passages are about different topics
      0.3 = tangentially related but wouldn't help answer the query
      0.5 = partially relevant, some useful information present
      0.7 = mostly relevant, good chance the answer is in the passages
      1.0 = highly relevant, the passages directly answer the query

    Respond with ONLY a float between 0.0 and 1.0. No explanation.
""")

_REWRITE_SYSTEM = textwrap.dedent("""\
    You are a search query optimizer.

    The original query failed to find relevant results. Rewrite it to be
    more likely to match relevant documents.

    Strategies:
    - Use different terminology or synonyms
    - Break it into a more atomic sub-question
    - Make it more general (remove overly specific constraints)
    - Rephrase as a keyword search rather than a full sentence

    Respond with ONLY the rewritten query. No explanation.
""")

_REFINE_SYSTEM = textwrap.dedent("""\
    You are a passage filter.

    Given a query and retrieved passages, extract ONLY the sentences that
    are directly relevant to answering the query. Remove irrelevant content.

    Return the filtered text — keep it factual and preserve exact wording.
    If nothing is relevant, return an empty string.
""")

# Relevance thresholds
_HIGH = 0.7
_LOW = 0.3


class CRAG(BaseRAG):
    """
    Corrective RAG: evaluate retrieval quality, then correct if needed.

    Parameters
    ----------
    max_retries : number of query rewrite attempts in the LOW path
    """

    def __init__(self, *args, max_retries: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._max_retries = max_retries

    def run(self, query: str) -> RAGResponse:
        # Stage 1: initial retrieval
        results = self._retrieve(query, k=config.top_k)

        if not results:
            return self.generator.generate(
                query=query, results=[], pattern="crag",
            )

        # Stage 2: relevance evaluation
        score = self._score_relevance(query, results)

        if score >= _HIGH:
            # HIGH — retrieved docs are good, proceed
            action = "proceed"

        elif score >= _LOW:
            # MEDIUM — refine: strip irrelevant sentences from chunks
            action = "refine"
            results = self._refine_results(query, results)

        else:
            # LOW — correct: rewrite query and re-retrieve
            action = "correct"
            results = self._correct(query, results)

        response = self.generator.generate(
            query=query,
            results=results,
            pattern="crag",
        )
        response.metadata["crag_relevance_score"] = score
        response.metadata["crag_action"] = action
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_relevance(self, query: str, results) -> float:
        """Ask Claude to rate how relevant the retrieved passages are."""
        passages = "\n\n---\n\n".join(r.chunk.content[:400] for r in results[:5])
        prompt = f"Query: {query}\n\nPassages:\n{passages}"
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=8,
            temperature=0,
            system=_RELEVANCE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return float(resp.content[0].text.strip())
        except ValueError:
            return 0.5  # neutral fallback

    def _refine_results(self, query: str, results):
        """
        Strip irrelevant sentences from each retrieved chunk.
        Returns refined results (fewer/shorter chunks).
        """
        from src.documents import Chunk, RetrievalResult
        import dataclasses

        refined = []
        for r in results:
            prompt = f"Query: {query}\n\nPassage:\n{r.chunk.content}"
            resp = self._client.messages.create(
                model=config.model,
                max_tokens=256,
                temperature=0,
                system=_REFINE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            filtered_text = resp.content[0].text.strip()
            if filtered_text:
                # Create a new chunk with the filtered content
                new_chunk = dataclasses.replace(
                    r.chunk,
                    content=filtered_text,
                )
                refined.append(RetrievalResult(chunk=new_chunk, score=r.score))

        return refined if refined else results

    def _correct(self, query: str, original_results):
        """
        Rewrite the query and re-retrieve. Falls back to original results
        if the rewrite also produces low-quality results.
        """
        rewritten = self._rewrite_query(query)
        new_results = self._retrieve(rewritten, k=config.top_k)

        if not new_results:
            return original_results

        # Accept rewritten results if they score higher
        new_score = self._score_relevance(query, new_results)
        orig_score = self._score_relevance(query, original_results)

        return new_results if new_score >= orig_score else original_results

    def _rewrite_query(self, query: str) -> str:
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=80,
            temperature=0.3,
            system=_REWRITE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        return resp.content[0].text.strip()
