"""
Structured Logging and Observability

Replaces ad-hoc print() statements with structured JSON log lines.
Each event is a dict with a fixed set of fields so logs can be
ingested by Datadog, CloudWatch, or any log aggregator.

Event types
-----------
  query_start   — user submitted a question
  retrieval     — chunks retrieved from vector store
  generation    — Claude returned an answer
  cache_hit     — response served from cache
  error         — something went wrong
  metrics       — faithfulness / relevance scores

Usage
-----
  logger = RAGLogger(name="rag.prod")
  logger.query_start(query="...", pattern="naive", user_id="u123")
  logger.retrieval(query="...", n_chunks=5, scores=[0.82, 0.71, ...], latency_ms=12)
  logger.generation(query="...", pattern="naive", latency_ms=1240,
                    input_tokens=820, output_tokens=310, cache_hit=False)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional


class RAGLogger:
    """Structured JSON logger for the RAG system."""

    def __init__(self, name: str = "rag"):
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._log.addHandler(handler)
            self._log.setLevel(logging.INFO)

    def _emit(self, event: str, **fields) -> None:
        record = {"event": event, "ts": time.time(), **fields}
        self._log.info(json.dumps(record))

    def query_start(
        self,
        query: str,
        pattern: str,
        user_id: Optional[str] = None,
    ) -> None:
        self._emit("query_start", query=query[:200], pattern=pattern,
                   user_id=user_id)

    def retrieval(
        self,
        query: str,
        n_chunks: int,
        scores: List[float],
        latency_ms: float,
        fallback: bool = False,
    ) -> None:
        self._emit(
            "retrieval",
            query=query[:200],
            n_chunks=n_chunks,
            top_score=round(scores[0], 4) if scores else 0,
            avg_score=round(sum(scores) / len(scores), 4) if scores else 0,
            latency_ms=round(latency_ms, 1),
            fallback=fallback,
        )

    def generation(
        self,
        query: str,
        pattern: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False,
        conflict: bool = False,
    ) -> None:
        self._emit(
            "generation",
            query=query[:200],
            pattern=pattern,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            conflict=conflict,
        )

    def metrics(
        self,
        query: str,
        faithfulness: Optional[float],
        context_precision: Optional[float],
        context_recall: Optional[float],
        answer_relevance: Optional[float],
    ) -> None:
        self._emit(
            "metrics",
            query=query[:200],
            faithfulness=round(faithfulness, 3) if faithfulness is not None else None,
            context_precision=round(context_precision, 3) if context_precision is not None else None,
            context_recall=round(context_recall, 3) if context_recall is not None else None,
            answer_relevance=round(answer_relevance, 3) if answer_relevance is not None else None,
        )

    def error(self, query: str, pattern: str, error: str) -> None:
        self._emit("error", query=query[:200], pattern=pattern, error=error)

    def cache_hit(self, query: str, pattern: str) -> None:
        self._emit("cache_hit", query=query[:200], pattern=pattern)


# Module-level singleton
logger = RAGLogger()
