"""
HyDE — Hypothetical Document Embeddings
(Gao et al., 2022 — https://arxiv.org/abs/2212.10496)

Core Idea
---------
The mismatch between short query vectors and long document vectors is a
fundamental problem for dense retrieval.  HyDE bridges the gap by:

  1. Asking the LLM to *hallucinate* a plausible document that would
     answer the query — the hypothetical document.
  2. Embedding the hypothetical document instead of the raw query.
  3. Searching with that richer embedding.

Because the hypothetical document uses the same vocabulary and writing
style as real documents, its embedding lives closer to real documents in
vector space than a short query does.

       query
         │
    [LLM: generate hypothetical doc]
         │ hyp_doc
    [Embedder]
         │ hyp_vec
    [VectorStore.search] ──→ top-k real chunks
         │
    [Generator] ──→ answer + citations  (using REAL chunks, not hyp_doc)

Note: The hypothetical document is discarded after embedding.
The final answer is grounded only in real retrieved passages.

When does HyDE help?
--------------------
- Rare or technical vocabulary gaps between query and corpus.
- Short queries that don't capture the full semantic scope of the answer.

When does it hurt?
------------------
- Highly factual queries where hallucinated content shifts the embedding
  away from the correct passage.
- Domains where the LLM lacks prior knowledge to produce a useful hypothesis.
"""

from __future__ import annotations

import textwrap

import anthropic

from src.config import config
from src.generation.generator import RAGResponse
from .base import BaseRAG


_HYDE_SYSTEM = textwrap.dedent("""\
    You are a document generation assistant.
    Write a short factual passage (2–4 sentences) that would directly and
    completely answer the given question, as if it were an excerpt from a
    high-quality reference document.
    Do NOT explain that you are writing a hypothetical passage.
    Write only the passage itself.
""")


class HyDERAG(BaseRAG):
    """
    HyDE: generate a hypothetical answer document, embed it, retrieve real docs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _generate_hypothesis(self, query: str) -> str:
        """Ask Claude to hallucinate an ideal answer passage."""
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=256,
            temperature=0.7,      # higher temp → more diverse vocabulary coverage
            system=_HYDE_SYSTEM,
            messages=[{"role": "user", "content": f"Question: {query}"}],
        )
        return resp.content[0].text.strip()

    def run(self, query: str) -> RAGResponse:
        # Step 1: generate hypothetical document
        hypothesis = self._generate_hypothesis(query)

        # Step 2: embed the hypothesis (not the raw query)
        hyp_vec = self.embedder.embed_query(hypothesis)

        # Step 3: search with hypothesis embedding
        results = self.store.search(
            hyp_vec,
            k=config.top_k,
            min_score=0.0,
        )

        # Step 4: generate answer from REAL retrieved docs
        response = self.generator.generate(
            query=query,
            results=results,
            pattern="hyde",
        )
        response.metadata["hypothesis"] = hypothesis
        return response
