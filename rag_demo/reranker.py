"""
reranker.py — Second-stage precision: re-rank the top-20 retrieval results down to top-5.

Two-stage retrieval:
    Stage 1 (recall):   BM25 + vector search → top-20 candidates
                        Fast: O(N) matrix multiply or inverted index lookup
    Stage 2 (precision): cross-encoder → top-5 final results
                         Slow but accurate: each (query, chunk) pair is re-scored

Why cross-encoder is better than bi-encoder for re-ranking:
    Bi-encoder (Stage 1):
        embed(query) → q_vec
        embed(chunk) → c_vec
        score = dot(q_vec, c_vec)
        ❌ Query and chunk never "see" each other during encoding.

    Cross-encoder (Stage 2):
        input = "[CLS] query [SEP] chunk [SEP]"
        BERT reads BOTH together → full cross-attention between every query word
        and every chunk word → much richer relevance signal.
        ❌ But O(candidates) forward passes → too slow for 10M chunks.
"""
import os
from dataclasses import dataclass
from vectorstore import RetrievalResult


class CrossEncoderReranker:
    """
    Uses a cross-encoder model (BERT-based) to score (query, chunk) pairs.

    The model outputs a single relevance score per pair.
    We sort by that score and return the top_k.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Lazy import — only loaded when cross-encoder is actually used
        from sentence_transformers import CrossEncoder
        # ms-marco-MiniLM-L-6-v2 is a small, fast model fine-tuned on MS MARCO
        # (Microsoft Machine Reading Comprehension dataset — 8.8M QA pairs).
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        # Build pairs: [(query, chunk1_text), (query, chunk2_text), ...]
        # The cross-encoder sees both together — this is the key difference
        # from bi-encoders where query and chunk are embedded separately.
        pairs = [(query, r.chunk.content) for r in results]

        # predict() runs one forward pass per pair through the cross-encoder.
        # Returns a list of floats (relevance scores, no fixed range).
        scores = self.model.predict(pairs)

        # Attach the new scores and sort
        rescored = [
            RetrievalResult(chunk=r.chunk, score=float(s))
            for r, s in zip(results, scores)
        ]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


class LLMReranker:
    """
    Uses Claude to score each (query, chunk) pair.
    Slower and more expensive than CrossEncoder but understands nuance better.
    Good for high-stakes queries where precision is critical.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        rescored: list[RetrievalResult] = []

        for r in results:
            prompt = (
                f"Rate how relevant this passage is for answering the query.\n"
                f"Query: {query}\n"
                f"Passage: {r.chunk.content[:500]}\n"
                f"Respond with only a number from 0 to 10."
            )
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}],
                )
                score = float(response.content[0].text.strip())
            except Exception:
                score = 0.0

            rescored.append(RetrievalResult(chunk=r.chunk, score=score))

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
