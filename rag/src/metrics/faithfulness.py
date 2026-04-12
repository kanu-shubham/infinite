"""
Faithfulness Metric (LLM-as-Judge)

Definition
----------
Faithfulness measures whether every factual claim in the generated answer
is directly supported by the retrieved context passages.

A score of 1.0 means every claim is grounded; 0.0 means none are.

Algorithm (RAGAS-inspired)
--------------------------
1. Ask Claude to extract all atomic claims from the answer.
   "Atomic" means one fact per statement, e.g.:
     "BERT uses bidirectional attention" → one claim
     "BERT was trained on Wikipedia and BookCorpus" → two claims

2. For each claim, ask Claude whether it is supported by the context.

3. Faithfulness = supported_claims / total_claims

Why LLM-as-judge?
-----------------
Rule-based NLI (Natural Language Inference) models require downloading
weights and have poor domain coverage.  LLM-as-judge generalises to any
domain and correlates well with human judgment (Zheng et al., 2023).

Production note
---------------
For high-throughput systems, run claim extraction + verification in a
single batched prompt to save API calls.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import List

import anthropic

from src.config import config
from src.generation.generator import RAGResponse


_EXTRACT_SYSTEM = textwrap.dedent("""\
    Extract every atomic factual claim from the text below.
    An atomic claim is a single, self-contained factual statement.
    Return a JSON array of strings. Example: ["Claim A.", "Claim B."]
    Return only the JSON array, no other text.
""")

_VERIFY_SYSTEM = textwrap.dedent("""\
    You are a strict fact-checker.
    Given a context and a claim, answer YES if the claim is DIRECTLY and
    EXPLICITLY supported by the context, or NO if it is not found or
    contradicted. Answer with YES or NO only.
""")


@dataclass
class FaithfulnessResult:
    score: float                # 0.0 – 1.0
    total_claims: int
    supported_claims: int
    unsupported: List[str]      # claims that failed verification

    def __str__(self):
        return (
            f"Faithfulness: {self.score:.2f} "
            f"({self.supported_claims}/{self.total_claims} claims supported)"
        )


class FaithfulnessEvaluator:
    """Evaluate how well an answer is grounded in its source passages."""

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _extract_claims(self, answer: str) -> List[str]:
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=512,
            temperature=0,
            system=_EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": answer}],
        )
        text = resp.content[0].text.strip()
        try:
            claims = json.loads(text)
            if isinstance(claims, list):
                return [str(c) for c in claims]
        except json.JSONDecodeError:
            pass
        # Fallback: split on newlines
        return [line.strip("- •*").strip() for line in text.splitlines() if line.strip()]

    def _verify_claim(self, claim: str, context: str) -> bool:
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=4,
            temperature=0,
            system=_VERIFY_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context[:2000]}\n\nClaim: {claim}",
            }],
        )
        return resp.content[0].text.strip().upper().startswith("Y")

    def evaluate(self, response: RAGResponse) -> FaithfulnessResult:
        """
        Compute faithfulness for a RAGResponse.

        Parameters
        ----------
        response : RAGResponse — must have .answer and .retrieval_results set.

        Returns
        -------
        FaithfulnessResult
        """
        if not response.retrieval_results:
            # No context → can't be grounded → score 0
            return FaithfulnessResult(
                score=0.0, total_claims=0,
                supported_claims=0, unsupported=[],
            )

        context = "\n\n".join(
            r.chunk.content for r in response.retrieval_results[:5]
        )
        claims = self._extract_claims(response.answer)

        if not claims:
            return FaithfulnessResult(
                score=1.0, total_claims=0,
                supported_claims=0, unsupported=[],
            )

        supported = 0
        unsupported = []
        for claim in claims:
            if self._verify_claim(claim, context):
                supported += 1
            else:
                unsupported.append(claim)

        score = supported / len(claims) if claims else 0.0
        return FaithfulnessResult(
            score=score,
            total_claims=len(claims),
            supported_claims=supported,
            unsupported=unsupported,
        )
