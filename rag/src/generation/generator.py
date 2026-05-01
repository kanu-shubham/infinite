"""
LLM-based response generator using Claude.

Responsibilities
----------------
1. Format retrieved chunks into a context block with citations.
2. Call Claude to produce a grounded answer.
3. Return a structured response with the answer, citations, and raw context.

Prompt caching
--------------
The system prompt and document context are marked with cache_control so that
repeated queries over the same corpus don't re-encode the context on every call,
cutting both latency and cost significantly.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import time

import anthropic

from src.config import config
from src.documents import Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retry(fn, max_retries: int = 4, base_delay: float = 2.0):
    """
    Exponential backoff retry for Anthropic API calls.

    Retries on:
      - RateLimitError  (429) — slow down and retry
      - APIStatusError  (500, 529) — server overloaded, retry
      - APIConnectionError — transient network failure

    Raises immediately on:
      - AuthenticationError (401) — bad API key, no point retrying
      - BadRequestError (400) — malformed request, won't improve
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            if e.status_code in (500, 529):
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise
        except anthropic.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    chunk_id: str
    doc_title: str
    doc_source: str
    excerpt: str          # First 120 chars of the chunk
    score: float


@dataclass
class RAGResponse:
    """Everything the RAG pipeline returns for one query."""
    query: str
    answer: str
    citations: List[Citation]
    retrieval_results: List[RetrievalResult]
    pattern: str = "naive"
    fallback: bool = False       # True when no relevant docs were found
    conflict_detected: bool = False
    metadata: dict = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [
            f"[{self.pattern.upper()} RAG]",
            f"Query : {self.query}",
            "",
            "Answer",
            "------",
            self.answer,
        ]
        if self.fallback:
            lines.append("\n[Note: answer generated without retrieved context — no relevant documents found]")
        if self.conflict_detected:
            lines.append("\n[Warning: conflicting information detected in sources]")
        if self.citations:
            lines.append("\nSources")
            lines.append("-------")
            for i, c in enumerate(self.citations, 1):
                lines.append(f"[{i}] {c.doc_title or c.chunk_id} ({c.doc_source or 'unknown'})")
                lines.append(f"    \"{c.excerpt}...\"")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise research assistant.  Your answers must be:

    1. GROUNDED — every factual claim must be directly supported by the
       provided context passages.  If a claim is not in the context, say so.
    2. CITED — reference the passage number [1], [2], etc. whenever you use
       information from it.
    3. HONEST — if the context passages do not contain enough information to
       answer the question, say "I don't have enough information to answer
       this confidently" and explain what is missing.
    4. CONCISE — one to three paragraphs unless the question requires more.

    Never hallucinate facts.  Never contradict the context passages.
""")

_CONFLICT_HINT = textwrap.dedent("""\

    IMPORTANT: The retrieved passages below contain conflicting information.
    Acknowledge the conflict in your answer and cite both sides.
""")


def _format_context(results: List[RetrievalResult]) -> str:
    """Build a numbered context block from retrieval results."""
    parts = []
    for i, r in enumerate(results, 1):
        title = r.chunk.doc_title or r.chunk.doc_id
        source = r.chunk.doc_source or ""
        header = f"[Passage {i}] {title}"
        if source:
            header += f" | {source}"
        header += f" | score={r.score:.3f}"
        parts.append(f"{header}\n{r.chunk.content}")
    return "\n\n".join(parts)


def _detect_conflict(results: List[RetrievalResult], client: anthropic.Anthropic) -> bool:
    """
    Ask Claude whether the retrieved passages contradict each other.
    Returns True if a conflict is detected.

    This is a lightweight heuristic check — two passages are "conflicting"
    only when they make clearly opposite factual claims.
    """
    if len(results) < 2:
        return False

    context = _format_context(results[:4])  # check top-4 only to save tokens
    resp = client.messages.create(
        model=config.model,
        max_tokens=16,
        temperature=0,
        system="Answer with YES or NO only.",
        messages=[{
            "role": "user",
            "content": (
                "Do the following passages contain any directly contradictory "
                "factual claims (not just different perspectives)?\n\n"
                f"{context}"
            ),
        }],
    )
    return resp.content[0].text.strip().upper().startswith("YES")


class Generator:
    """Wraps the Anthropic client and handles all generation logic."""

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def generate(
        self,
        query: str,
        results: List[RetrievalResult],
        pattern: str = "naive",
        check_conflict: bool = True,
        min_score: float | None = None,
    ) -> RAGResponse:
        """
        Generate a grounded answer for *query* given *results*.

        Parameters
        ----------
        query          : the user's question
        results        : retrieved chunks with similarity scores
        pattern        : name of the RAG strategy (for logging/display)
        check_conflict : whether to run the conflict-detection check
        min_score      : override config.min_similarity for this call
        """
        threshold = min_score if min_score is not None else config.min_similarity

        # ---- No-result fallback ----
        valid_results = [r for r in results if r.score >= threshold]
        fallback = len(valid_results) == 0

        # ---- Conflict detection ----
        conflict = False
        if not fallback and check_conflict and len(valid_results) >= 2:
            conflict = _detect_conflict(valid_results, self._client)

        # ---- Build prompt ----
        if fallback:
            user_content = (
                f"Question: {query}\n\n"
                "No relevant documents were retrieved.  Please answer from "
                "general knowledge if possible, or state what you don't know."
            )
            system = _SYSTEM_PROMPT
        else:
            context_block = _format_context(valid_results)
            conflict_hint = _CONFLICT_HINT if conflict else ""
            system = _SYSTEM_PROMPT + conflict_hint
            user_content = (
                f"Context passages:\n\n{context_block}\n\n"
                f"Question: {query}"
            )

        # ---- Call Claude with prompt caching ----
        # The system prompt and context are stable across hops → cache them.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        response = _with_retry(lambda: self._client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        ))

        answer = response.content[0].text.strip()

        # ---- Build citation objects ----
        citations = [
            Citation(
                chunk_id=r.chunk.chunk_id,
                doc_title=r.chunk.doc_title,
                doc_source=r.chunk.doc_source,
                excerpt=r.chunk.content[:120],
                score=r.score,
            )
            for r in valid_results
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieval_results=valid_results,
            pattern=pattern,
            fallback=fallback,
            conflict_detected=conflict,
            metadata={"input_tokens": response.usage.input_tokens,
                      "output_tokens": response.usage.output_tokens},
        )
