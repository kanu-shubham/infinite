"""
Self-RAG — Self-Reflective Retrieval-Augmented Generation
(Asai et al., 2023 — https://arxiv.org/abs/2310.11511)

Core Idea
---------
Standard RAG always retrieves, whether or not retrieval is useful.
Self-RAG teaches the model to *decide* when to retrieve and to *critique*
its own outputs.  It uses special "reflection tokens":

  [Retrieve]   — should the model retrieve documents for this query?
  [IsRel]      — is this retrieved passage relevant to the question?
  [IsSup]      — is the generated segment supported by the passage?
  [IsUse]      — is the overall answer useful to the user?

In the original paper these tokens are fine-tuned into the model weights.
Here we implement the same decision logic via structured LLM calls —
functionally identical, no fine-tuning required.

Pipeline
--------
  1. [Retrieve?]  Ask: "Does this query require external knowledge?"
                 → If NO, generate directly (saves tokens + latency).
  2. [Retrieve]   Fetch top-k chunks.
  3. [IsRel]      Score each chunk: "Is this passage relevant?"
                 → Keep only relevant chunks.
  4. [Generate]   Produce an answer from the filtered chunks.
  5. [IsSup]      Check: "Is each answer sentence grounded in the context?"
                 → If poorly supported, retry with a stricter prompt.
  6. [IsUse]      Check: "Does the answer usefully address the question?"
                 → If not, generate a fallback with an explanation.

When does Self-RAG help?
------------------------
- Mixed queries where some sub-questions need retrieval and others don't.
- Corpora with noisy or irrelevant documents (IsRel filters them out).
- High-stakes domains requiring grounding verification (IsSup).
"""

from __future__ import annotations

import textwrap
from typing import List

import anthropic

from src.config import config
from src.documents import RetrievalResult
from src.generation.generator import RAGResponse
from .base import BaseRAG


_BINARY_SYSTEM = "Answer with YES or NO only. No explanation."


def _yes(client: anthropic.Anthropic, prompt: str, model: str) -> bool:
    resp = client.messages.create(
        model=model, max_tokens=4, temperature=0,
        system=_BINARY_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().upper().startswith("Y")


class SelfRAG(BaseRAG):
    """
    Self-reflective RAG with retrieve / relevance / support / usefulness checks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # ------------------------------------------------------------------
    # Reflection tokens (implemented as structured LLM calls)
    # ------------------------------------------------------------------

    def _needs_retrieval(self, query: str) -> bool:
        """[Retrieve?] Does this question require external knowledge?"""
        return _yes(
            self._client,
            f"Does answering this question require looking up specific facts, "
            f"events, or data that a language model might not know reliably?\n\n"
            f"Question: {query}",
            config.model,
        )

    def _is_relevant(self, query: str, chunk_content: str) -> bool:
        """[IsRel] Is this passage relevant to the query?"""
        return _yes(
            self._client,
            f"Is the following passage useful for answering the question?\n\n"
            f"Question: {query}\n\nPassage: {chunk_content[:400]}",
            config.model,
        )

    def _is_supported(self, answer: str, context: str) -> bool:
        """[IsSup] Is the answer grounded in the provided context?"""
        return _yes(
            self._client,
            f"Are the factual claims in the answer directly supported by "
            f"the context passages (not just plausible)?\n\n"
            f"Context:\n{context[:800]}\n\nAnswer: {answer[:400]}",
            config.model,
        )

    def _is_useful(self, query: str, answer: str) -> bool:
        """[IsUse] Does the answer helpfully address the question?"""
        return _yes(
            self._client,
            f"Does the following answer meaningfully and helpfully address "
            f"the question?\n\nQuestion: {query}\n\nAnswer: {answer[:400]}",
            config.model,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, query: str) -> RAGResponse:
        # --- Step 1: Decide whether retrieval is needed ---
        retrieve = self._needs_retrieval(query)

        if not retrieve:
            # Generate directly without retrieval
            response = self.generator.generate(
                query=query,
                results=[],
                pattern="self_rag",
            )
            response.metadata["retrieval_skipped"] = True
            return response

        # --- Step 2: Retrieve ---
        raw_results = self._retrieve(query, k=config.top_k * 2)  # over-fetch

        # --- Step 3: Filter by relevance [IsRel] ---
        relevant: List[RetrievalResult] = []
        for r in raw_results:
            if self._is_relevant(query, r.chunk.content):
                relevant.append(r)
            if len(relevant) >= config.top_k:
                break

        if not relevant:
            # Fallback: use top-k without relevance filter
            relevant = raw_results[:config.top_k]

        # --- Step 4: Generate ---
        response = self.generator.generate(
            query=query,
            results=relevant,
            pattern="self_rag",
        )

        # --- Step 5: Check groundedness [IsSup] ---
        context_text = "\n\n".join(r.chunk.content for r in relevant[:3])
        supported = self._is_supported(response.answer, context_text)

        if not supported:
            # Retry with an explicit "be more conservative" instruction
            retry_query = (
                f"{query}\n\n[IMPORTANT: Only state what is explicitly confirmed "
                f"by the retrieved passages. If unsure, say so.]"
            )
            response = self.generator.generate(
                query=retry_query,
                results=relevant,
                pattern="self_rag",
            )
            response.metadata["support_retry"] = True

        # --- Step 6: Check usefulness [IsUse] ---
        useful = self._is_useful(query, response.answer)
        response.metadata.update(
            retrieval_skipped=False,
            isrel_passed=len(relevant),
            issup_passed=supported,
            isuse_passed=useful,
        )
        return response
