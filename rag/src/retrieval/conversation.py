"""
Conversation-Aware Query Rewriter

The single biggest RAG failure in chat applications: pronouns and
references to previous messages.

  Turn 1: "Tell me about BERT."
           → retrieves correctly
  Turn 2: "How does it compare to GPT?"
           → "it" has no referent → retrieves nothing relevant

The fix: before retrieval, rewrite the query using conversation history
to make it self-contained.

  "How does it compare to GPT?"  +  history  →  "How does BERT compare to GPT?"

This runs one cheap LLM call before the main retrieval and pays for itself
by preventing expensive retrieval failures.

Usage
-----
  rewriter = ConversationRewriter()

  history = [
      {"role": "user",      "content": "Tell me about BERT."},
      {"role": "assistant", "content": "BERT is a transformer model..."},
  ]

  standalone = rewriter.rewrite("How does it compare to GPT?", history)
  # → "How does BERT compare to GPT?"
"""

from __future__ import annotations

import textwrap
from typing import List, Dict

import anthropic

from src.config import config


_REWRITE_SYSTEM = textwrap.dedent("""\
    You are a query rewriter for a search system.

    Given a conversation history and a follow-up question, rewrite the
    follow-up question to be completely self-contained — it should make
    sense with NO context from the conversation history.

    Rules:
    - Replace pronouns (it, they, this, that, these) with their referents.
    - Expand abbreviations if clarified earlier in the conversation.
    - Keep the rewritten question concise and search-friendly.
    - If the question is already self-contained, return it unchanged.
    - Return ONLY the rewritten question, no explanation.

    Examples:
    History: "User asked about BERT. Assistant explained BERT."
    Follow-up: "How does it compare to GPT?"
    Rewritten: "How does BERT compare to GPT?"

    History: "User asked about the pricing decision."
    Follow-up: "When was that decided?"
    Rewritten: "When was the pricing decision made?"
""")


class ConversationRewriter:
    """
    Rewrites a follow-up query to be standalone using conversation history.
    """

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def rewrite(
        self,
        query: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Parameters
        ----------
        query   : the user's latest message (may contain pronouns)
        history : list of {"role": "user"/"assistant", "content": "..."} dicts

        Returns
        -------
        A standalone version of the query, or the original if no rewrite needed.
        """
        if not history:
            return query

        # Check if rewrite is likely needed (cheap heuristic before LLM call)
        pronouns = {"it", "its", "this", "that", "they", "their", "these",
                    "those", "he", "she", "him", "her", "we", "our"}
        query_words = set(query.lower().split())
        if not (query_words & pronouns) and "?" not in query[:20]:
            # No pronouns detected — likely already standalone
            return query

        # Format history for the prompt (last 6 turns max)
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:300]}"
            for m in history[-6:]
        )

        prompt = f"Conversation history:\n{history_text}\n\nFollow-up question: {query}"

        resp = self._client.messages.create(
            model=config.model,
            max_tokens=128,
            temperature=0,
            system=_REWRITE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = resp.content[0].text.strip()
        # Safety: if rewriter returns something very long or empty, fall back
        if not rewritten or len(rewritten) > len(query) * 3:
            return query
        return rewritten
