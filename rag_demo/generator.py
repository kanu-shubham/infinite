"""
generator.py — Call Claude to generate an answer from retrieved context.
"""
import os
import time
from dataclasses import dataclass, field
from vectorstore import RetrievalResult


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieval_results: list[RetrievalResult]
    sources: list[str] = field(default_factory=list)
    pattern: str = "naive"


class Generator:

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_context_chunks: int = 5):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model
        self.max_context_chunks = max_context_chunks

    def generate(self, query: str, results: list[RetrievalResult], pattern: str = "naive") -> RAGResponse:
        # Take the top chunks as context
        top = results[: self.max_context_chunks]

        # Build context block
        context_parts = []
        sources = []
        for i, r in enumerate(top, 1):
            context_parts.append(f"[{i}] (source: {r.chunk.doc_source})\n{r.chunk.content}")
            if r.chunk.doc_source not in sources:
                sources.append(r.chunk.doc_source)

        context = "\n\n".join(context_parts)

        prompt = (
            f"Answer the question using ONLY the context below. "
            f"If the context does not contain the answer, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        answer = self._call_with_retry(prompt)
        return RAGResponse(
            query=query,
            answer=answer,
            retrieval_results=top,
            sources=sources,
            pattern=pattern,
        )

    def _call_with_retry(self, prompt: str, max_retries: int = 4) -> str:
        """
        Exponential backoff retry: 2s, 4s, 8s, 16s.
        Handles transient API errors (rate limits, 500s).
        """
        delay = 2
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error generating answer: {e}"
                time.sleep(delay)
                delay *= 2   # 2 → 4 → 8 → 16
        return "Failed to generate answer."
