"""
Multi-hop Retrieval
(inspired by HotpotQA / Iterative Retrieval Augmentation)

Core Idea
---------
Some questions cannot be answered with a single retrieval step because the
answer depends on *chaining* facts across documents:

  Q: "Which optimiser was used by the model that achieved the highest BLEU
      score on WMT14 English-French?"

  Hop 1: retrieve docs about BLEU scores on WMT14 → find the model name.
  Hop 2: retrieve docs about that model → find its optimiser.

Multi-hop retrieval iterates:
  1. Retrieve with the current query.
  2. Extract bridging entities / facts from retrieved docs.
  3. Formulate a new query that goes one step deeper.
  4. Repeat for max_hops steps.
  5. Generate the final answer from the union of all retrieved docs.

Pipeline
--------
  query_0
    │
    ├── [Retrieve] ──→ docs_0
    │     │
    │   [LLM: extract bridging info + formulate query_1]
    │
  query_1
    │
    ├── [Retrieve] ──→ docs_1
    │     │
    │   [LLM: extract bridging info + formulate query_2]
    │
  query_2  …
    │
    └── [Generate] ←── union(docs_0, docs_1, docs_2)

When does multi-hop help?
-------------------------
- Complex, multi-step reasoning questions.
- Knowledge graphs where facts are scattered across many documents.
- Questions with implicit sub-questions ("… of the person who …").
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional

import anthropic

from src.config import config
from src.documents import RetrievalResult
from src.generation.generator import RAGResponse
from .base import BaseRAG


_HOP_SYSTEM = textwrap.dedent("""\
    You are a research assistant performing iterative document retrieval.

    Given:
    - The original question
    - The passages retrieved so far
    - What has been learned so far

    Your task:
    1. Extract the key facts or entities from the retrieved passages that are
       needed to answer the original question.
    2. Identify what information is STILL MISSING to fully answer the question.
    3. Write a precise follow-up search query that would find the missing info.
       The query should be specific and different from the original question.

    Respond in this exact format:
    LEARNED: <one sentence summary of key facts found>
    MISSING: <what is still needed>
    NEXT_QUERY: <the follow-up search query, or "DONE" if the question can now be answered>
""")


def _parse_hop_response(text: str) -> dict:
    """Parse the structured hop response from the LLM."""
    result = {"learned": "", "missing": "", "next_query": "DONE"}
    for line in text.splitlines():
        if line.startswith("LEARNED:"):
            result["learned"] = line[len("LEARNED:"):].strip()
        elif line.startswith("MISSING:"):
            result["missing"] = line[len("MISSING:"):].strip()
        elif line.startswith("NEXT_QUERY:"):
            result["next_query"] = line[len("NEXT_QUERY:"):].strip()
    return result


class MultiHopRAG(BaseRAG):
    """
    Iterative retrieval: each hop refines the query based on what was found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _plan_next_hop(
        self,
        original_query: str,
        all_results: List[RetrievalResult],
        hop_log: List[dict],
    ) -> dict:
        """Ask Claude what to search for next."""
        context_summary = "\n\n".join(
            f"[Passage {i+1}] {r.chunk.content[:300]}"
            for i, r in enumerate(all_results[:6])
        )
        learned_so_far = " ".join(h.get("learned", "") for h in hop_log)

        prompt = (
            f"Original question: {original_query}\n\n"
            f"Retrieved passages so far:\n{context_summary}\n\n"
            f"What has been learned: {learned_so_far or 'Nothing yet.'}"
        )

        resp = self._client.messages.create(
            model=config.model,
            max_tokens=200,
            temperature=0,
            system=_HOP_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_hop_response(resp.content[0].text)

    def run(self, query: str) -> RAGResponse:
        all_results: List[RetrievalResult] = []
        seen_chunk_ids: set = set()
        hop_log: List[dict] = []
        current_query = query

        for hop in range(config.max_hops):
            # Retrieve with current query
            hop_results = self._retrieve(current_query, k=config.top_k)

            # Deduplicate
            new_results = [
                r for r in hop_results
                if r.chunk.chunk_id not in seen_chunk_ids
            ]
            for r in new_results:
                seen_chunk_ids.add(r.chunk.chunk_id)
            all_results.extend(new_results)

            # Plan next hop
            hop_info = self._plan_next_hop(query, all_results, hop_log)
            hop_log.append(hop_info)

            next_query = hop_info.get("next_query", "DONE")
            if next_query == "DONE" or not next_query:
                break

            current_query = next_query

        # Generate from the union of all retrieved docs, ranked by score
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:config.top_k]

        response = self.generator.generate(
            query=query,
            results=top_results,
            pattern="multi_hop",
        )
        response.metadata["hops"] = len(hop_log)
        response.metadata["hop_log"] = hop_log
        return response
