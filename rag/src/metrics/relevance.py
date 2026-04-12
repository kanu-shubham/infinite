"""
Relevance Metrics (LLM-as-Judge)

Three distinct relevance signals, following the RAGAS framework:

1. Context Precision  — Of the retrieved passages, what fraction is relevant?
   Formula: relevant_passages / total_passages
   Penalises retrieving noisy, off-topic chunks.

2. Context Recall     — Of the information needed to answer the query, what
   fraction is covered by the retrieved passages?
   Formula: covered_aspects / total_aspects_needed
   Penalises missing critical information.

3. Answer Relevance   — Does the answer actually address the question?
   Uses a generative proxy: if we embed generated follow-up questions
   from the answer and compare to the original query, high similarity
   → the answer is on-topic.  Here we use an LLM judge instead.

These three metrics together form the retrieval quality assessment:

  ┌──────────────────────────────────────────────────────────────────┐
  │  Context Precision ←→ retrieving the RIGHT documents             │
  │  Context Recall    ←→ retrieving ENOUGH documents                │
  │  Answer Relevance  ←→ the answer actually ANSWERS the question   │
  └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import List

import anthropic

from src.config import config
from src.generation.generator import RAGResponse


_BINARY = "Answer with YES or NO only."

_RECALL_SYSTEM = textwrap.dedent("""\
    Given a question and a set of context passages, identify the key aspects
    that need to be covered to fully answer the question.
    Then check which aspects are covered by the passages.

    Respond in this exact format:
    ASPECTS: <comma-separated list of required aspects>
    COVERED: <comma-separated list of aspects that ARE covered>
    SCORE: <fraction like 3/5>
""")

_ANSWER_RELEVANCE_SYSTEM = textwrap.dedent("""\
    Rate how well the answer addresses the question on a scale of 0–5:
    0 = completely off-topic
    3 = partially addresses the question
    5 = fully and precisely addresses the question

    Respond with only the integer score (0, 1, 2, 3, 4, or 5).
""")


@dataclass
class RelevanceResult:
    context_precision: float    # 0–1
    context_recall: float       # 0–1
    answer_relevance: float     # 0–1

    @property
    def overall(self) -> float:
        """Harmonic mean of all three metrics (penalises low scores on any axis)."""
        vals = [self.context_precision, self.context_recall, self.answer_relevance]
        if any(v == 0 for v in vals):
            return 0.0
        return len(vals) / sum(1.0 / v for v in vals)

    def __str__(self):
        return (
            f"Context Precision: {self.context_precision:.2f} | "
            f"Context Recall: {self.context_recall:.2f} | "
            f"Answer Relevance: {self.answer_relevance:.2f} | "
            f"Overall: {self.overall:.2f}"
        )


class RelevanceEvaluator:
    """Compute context precision, context recall, and answer relevance."""

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # ------------------------------------------------------------------
    # Context Precision
    # ------------------------------------------------------------------

    def _is_relevant_passage(self, query: str, passage: str) -> bool:
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=4,
            temperature=0,
            system=_BINARY,
            messages=[{
                "role": "user",
                "content": (
                    f"Is the following passage useful for answering the question?\n\n"
                    f"Question: {query}\n\nPassage: {passage[:400]}"
                ),
            }],
        )
        return resp.content[0].text.strip().upper().startswith("Y")

    def _context_precision(
        self, query: str, passages: List[str]
    ) -> float:
        if not passages:
            return 0.0
        relevant = sum(
            self._is_relevant_passage(query, p) for p in passages
        )
        return relevant / len(passages)

    # ------------------------------------------------------------------
    # Context Recall
    # ------------------------------------------------------------------

    def _context_recall(self, query: str, context: str) -> float:
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=200,
            temperature=0,
            system=_RECALL_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context[:2000]}",
            }],
        )
        text = resp.content[0].text.strip()
        # Parse SCORE: N/M
        for line in text.splitlines():
            if line.startswith("SCORE:"):
                fraction = line[len("SCORE:"):].strip()
                try:
                    num, denom = fraction.split("/")
                    denom_int = int(denom.strip())
                    if denom_int == 0:
                        return 0.0
                    return min(1.0, int(num.strip()) / denom_int)
                except (ValueError, ZeroDivisionError):
                    pass
        return 0.5  # neutral fallback

    # ------------------------------------------------------------------
    # Answer Relevance
    # ------------------------------------------------------------------

    def _answer_relevance(self, query: str, answer: str) -> float:
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=4,
            temperature=0,
            system=_ANSWER_RELEVANCE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Question: {query}\n\nAnswer: {answer[:600]}",
            }],
        )
        text = resp.content[0].text.strip()
        try:
            score = int(text[0])  # first digit
            return min(1.0, score / 5.0)
        except (ValueError, IndexError):
            return 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, response: RAGResponse) -> RelevanceResult:
        """
        Compute all three relevance metrics for a RAGResponse.
        """
        query = response.query
        passages = [r.chunk.content for r in response.retrieval_results]
        context = "\n\n".join(passages[:5])

        precision = self._context_precision(query, passages)
        recall = self._context_recall(query, context) if context else 0.0
        relevance = self._answer_relevance(query, response.answer)

        return RelevanceResult(
            context_precision=precision,
            context_recall=recall,
            answer_relevance=relevance,
        )
