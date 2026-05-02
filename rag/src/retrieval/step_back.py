"""
Step-Back Prompting  (Google DeepMind, 2023)
arxiv.org/abs/2310.06117

The Idea
--------
Some questions are too specific for good retrieval.  "What caused the
2008 financial crisis specifically in September?" is hard to retrieve for
because no single chunk likely contains that exact framing.

Step-back prompting first asks: "What are the general principles of
financial crises?" — a higher-level question.  It retrieves context for
BOTH the abstract question AND the original, then answers the original
with combined context.

Two-stage pipeline
------------------
  Original question: "What is the boiling point of ethanol at 3000m altitude?"
  Step-back question: "How does altitude affect the boiling point of liquids?"

  Retrieve for both → combine → answer original with full context.

When it helps
-------------
- Physics / scientific questions requiring background principles
- History questions needing context about an era
- Technical questions where you need general concepts first
- Any question where the answer requires domain reasoning, not just lookup

When NOT to use
---------------
- Simple factual lookups ("Who wrote Hamlet?")
- Questions where the step-back would lose important specifics
"""

from __future__ import annotations

import textwrap
from typing import List

import anthropic

from src.config import config
from src.documents import RetrievalResult
from src.generation.generator import RAGResponse
from src.retrieval.base import BaseRAG


_STEP_BACK_SYSTEM = textwrap.dedent("""\
    You are a search query optimizer.

    Given a specific question, generate a broader "step-back" question that
    asks about the general principle, concept, or background knowledge needed
    to answer the original.

    Examples:
      Original: "What was Einstein's role at Princeton in 1950?"
      Step-back: "What was Einstein's career and academic positions?"

      Original: "What is the boiling point of ethanol at 3000m altitude?"
      Step-back: "How does altitude affect the boiling point of liquids?"

      Original: "How did BERT improve on ELMo for NLP tasks?"
      Step-back: "What are the key approaches to contextual word embeddings in NLP?"

    Respond with ONLY the step-back question. No explanation.
""")


class StepBackRAG(BaseRAG):
    """
    RAG with step-back prompting for principle-heavy questions.

    Retrieves for both the original query and a more abstract step-back
    version, then generates an answer grounded in the combined context.

    Parameters
    ----------
    n_original : top-k chunks for the original query
    n_stepback : top-k chunks for the step-back query
    """

    def __init__(self, *args, n_original: int = 3, n_stepback: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._n_original = n_original
        self._n_stepback = n_stepback

    def run(self, query: str) -> RAGResponse:
        # Step 1 — generate the step-back question
        step_back_query = self._generate_step_back(query)

        # Step 2 — retrieve for both queries
        original_results = self._retrieve(query, k=self._n_original)
        stepback_results = self._retrieve(step_back_query, k=self._n_stepback)

        # Step 3 — deduplicate by chunk_id, original results take priority
        seen: set = set()
        combined: List[RetrievalResult] = []
        for r in original_results + stepback_results:
            if r.chunk.chunk_id not in seen:
                seen.add(r.chunk.chunk_id)
                combined.append(r)

        # Step 4 — generate with combined context
        response = self.generator.generate(
            query=query,
            results=combined,
            pattern="step_back",
        )
        response.metadata["step_back_query"] = step_back_query
        response.metadata["n_original_chunks"] = len(original_results)
        response.metadata["n_stepback_chunks"] = len(stepback_results)
        return response

    def _generate_step_back(self, query: str) -> str:
        """Ask Claude for the higher-level principle question."""
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=80,
            temperature=0,
            system=_STEP_BACK_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        return resp.content[0].text.strip()
