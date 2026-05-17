"""
cache.py — Three-level caching for the RAG pipeline.

Level 1: QueryCache      — cache full answers by (user_id, query)
Level 2: EmbeddingCache  — cache vectors by text hash
Level 3: PromptCache     — Claude server-side caching via cache_control

All use in-memory dicts here. In production: Redis for levels 1 and 2.
"""
import hashlib
import time
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — Query Cache
# ─────────────────────────────────────────────────────────────────────────────

class QueryCache:
    """
    Cache full RAG answers keyed by (user_id + query).

    Why include user_id in the key?
    Alice asking "what is the leave policy?" can see HR docs.
    Bob asking the same query cannot see HR docs.
    Same query → different answers → must cache separately per user.

    TTL (time to live): after this many seconds, the cached entry expires.
    Why? Documents get updated. Stale answers are dangerous.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        # _store: cache_key → {"answer": ..., "sources": ..., "expires_at": ...}
        self._store: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, user_id: str, query: str) -> str:
        # Normalise query: lowercase, strip whitespace
        # "Leave Policy?" and "leave policy?" → same cache key
        normalised = query.lower().strip()
        raw = f"{user_id}::{normalised}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, user_id: str, query: str) -> Optional[dict]:
        key = self._make_key(user_id, query)
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check if expired
        if time.time() > entry["expires_at"]:
            del self._store[key]   # clean up expired entry
            self._misses += 1
            return None

        self._hits += 1
        return entry["data"]

    def set(self, user_id: str, query: str, data: dict) -> None:
        key = self._make_key(user_id, query)
        self._store[key] = {
            "data": data,
            "expires_at": time.time() + self.ttl,
        }

    def invalidate_all(self) -> None:
        """
        Call this after re-indexing documents.
        All cached answers may now be stale.
        In production: tag cache entries by doc_id and only invalidate
        entries that used the updated document.
        """
        self._store.clear()

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "entries": len(self._store),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — Embedding Cache
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Cache embedding vectors keyed by text hash.

    Same text always → same embedding (deterministic).
    So caching is safe and correct.

    Saves:
    - Re-indexing: same chunk text appears again → skip embedding
    - Popular query terms: "leave policy" asked 1000 times → embed once
    """

    def __init__(self):
        # key: md5(text) → value: numpy array
        self._store: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._make_key(text)
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
            return result
        self._misses += 1
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self._make_key(text)
        self._store[key] = embedding

    def get_or_compute(self, text: str, embedder) -> np.ndarray:
        """
        Try cache first. On miss, compute and store.
        This is the pattern you use in practice.
        """
        cached = self.get(text)
        if cached is not None:
            return cached

        embedding = embedder.embed_query(text)
        self.set(text, embedding)
        return embedding

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "entries": len(self._store),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — Claude Prompt Cache (server-side)
# ─────────────────────────────────────────────────────────────────────────────

class CachedGenerator:
    """
    Uses Claude's prompt caching feature.

    How it works:
      - You mark parts of the prompt with cache_control: ephemeral
      - First request: Claude processes those tokens normally, stores them server-side
      - Subsequent requests with same cached content: Claude skips re-processing
      - Cache read tokens cost 10% of normal input tokens → 90% savings

    Best used for:
      - Large system prompts (never change)
      - Context that is reused across many queries (popular documents)
      - Few-shot examples

    Cache lifetime: ~5 minutes of inactivity (ephemeral).
    If not called for 5 minutes, cache is evicted.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic, os
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model
        self._total_input_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_write_tokens = 0

    def generate(self, query: str, context_chunks: list[str]) -> str:
        """
        Generate an answer with prompt caching enabled.

        The context is marked as cacheable.
        The query is NOT cached (it changes every request).
        """
        context = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant. "
                        "Answer questions using ONLY the provided context. "
                        "If the answer is not in the context, say 'I don't know'."
                    ),
                    # Cache the system prompt — it never changes
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # Cache the context — may be reused across queries
                            "text": f"Context:\n{context}",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            # Do NOT cache the query — it changes every time
                            "text": f"\nQuestion: {query}",
                        },
                    ],
                }
            ],
        )

        # Track token usage to measure cache savings
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)
        self._total_cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0)

        return response.content[0].text

    @property
    def cache_savings_report(self) -> dict:
        """
        Shows how much money you saved using prompt caching.
        Prices per million tokens (Claude Sonnet):
          Normal input:  $3.00
          Cache write:   $3.75  (slightly more than normal to write)
          Cache read:    $0.30  (10% of normal — the big saving)
        """
        normal_cost = self._total_input_tokens * 3.00 / 1_000_000
        read_cost   = self._total_cache_read_tokens * 0.30 / 1_000_000
        write_cost  = self._total_cache_write_tokens * 3.75 / 1_000_000
        actual_cost = normal_cost + read_cost + write_cost

        hypothetical_cost = (
            self._total_input_tokens +
            self._total_cache_read_tokens +
            self._total_cache_write_tokens
        ) * 3.00 / 1_000_000

        return {
            "total_input_tokens":       self._total_input_tokens,
            "cache_read_tokens":        self._total_cache_read_tokens,
            "cache_write_tokens":       self._total_cache_write_tokens,
            "actual_cost_usd":          f"${actual_cost:.4f}",
            "without_cache_cost_usd":   f"${hypothetical_cost:.4f}",
            "savings_usd":              f"${hypothetical_cost - actual_cost:.4f}",
        }
