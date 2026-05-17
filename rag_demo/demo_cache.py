"""
demo_cache.py — Shows all three levels of caching working.
No API key needed — we mock the generator.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from cache import QueryCache, EmbeddingCache
from embeddings import TFIDFEmbedder


def main():
    print("=" * 60)
    print("Caching Demo — Three Levels")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────
    embedder = TFIDFEmbedder(n_components=8)
    embedder.fit([
        "employees get annual leave",
        "leave must be approved by manager",
        "sick leave requires doctor note",
        "travel requires manager approval",
    ])

    query_cache    = QueryCache(ttl_seconds=10)   # 10s TTL for demo
    embedding_cache = EmbeddingCache()

    # ── Level 2: Embedding Cache ───────────────────────────────────────
    print("\n[LEVEL 2 — Embedding Cache]")
    print("-" * 40)

    query = "how many leave days?"

    t0 = time.perf_counter()
    emb1 = embedding_cache.get_or_compute(query, embedder)
    t1 = time.perf_counter()
    print(f"  First call  (MISS): {(t1-t0)*1000:.2f}ms — computed embedding")

    t0 = time.perf_counter()
    emb2 = embedding_cache.get_or_compute(query, embedder)
    t1 = time.perf_counter()
    print(f"  Second call (HIT):  {(t1-t0)*1000:.2f}ms — returned from cache")

    print(f"  Same vector returned: {(emb1 == emb2).all()}")
    print(f"  Embedding cache stats: {embedding_cache.stats}")

    # ── Level 1: Query Cache ───────────────────────────────────────────
    print("\n[LEVEL 1 — Query Cache]")
    print("-" * 40)

    alice_id = "alice"
    bob_id   = "bob"

    # Mock answer that would come from the full pipeline
    mock_answer = {
        "answer": "Employees are entitled to 20 days of annual leave per year.",
        "sources": ["confluence/hr/leave-policy"],
    }

    # Alice asks the query — cache miss, full pipeline runs
    result = query_cache.get(alice_id, query)
    print(f"\n  Alice first query  → cache {'HIT' if result else 'MISS'}")
    if not result:
        print(f"  Running full pipeline for alice...")
        query_cache.set(alice_id, query, mock_answer)
        print(f"  Stored in cache.")

    # Alice asks same query again — cache hit
    result = query_cache.get(alice_id, query)
    print(f"\n  Alice same query   → cache {'HIT' if result else 'MISS'}")
    if result:
        print(f"  Returned from cache: '{result['answer'][:60]}...'")

    # Bob asks same query — cache MISS because different user_id
    result = query_cache.get(bob_id, query)
    print(f"\n  Bob same query     → cache {'HIT' if result else 'MISS'}")
    print(f"  (different user = different permissions = separate cache entry)")

    print(f"\n  Query cache stats: {query_cache.stats}")

    # ── TTL Expiry ─────────────────────────────────────────────────────
    print("\n[TTL EXPIRY — showing stale cache invalidation]")
    print("-" * 40)
    print(f"  Waiting 11 seconds for TTL=10s to expire...")
    time.sleep(11)
    result = query_cache.get(alice_id, query)
    print(f"  Alice query after TTL expired → cache {'HIT' if result else 'MISS (expired)'}")

    # ── Manual invalidation ────────────────────────────────────────────
    print("\n[MANUAL INVALIDATION — after document update]")
    print("-" * 40)

    # Re-cache first
    query_cache.set(alice_id, query, mock_answer)
    result = query_cache.get(alice_id, query)
    print(f"  Cache populated → {'HIT' if result else 'MISS'}")

    # Document updated → must clear cache
    print(f"  HR updates leave policy → invalidating all cache entries...")
    query_cache.invalidate_all()

    result = query_cache.get(alice_id, query)
    print(f"  After invalidation → cache {'HIT' if result else 'MISS (invalidated)'}")
    print(f"  Next query will use fresh data from updated document ✅")

    # ── Level 3: Prompt Cache explanation ─────────────────────────────
    print("\n[LEVEL 3 — Claude Prompt Cache]")
    print("-" * 40)
    print("""
  How it works in code:

  response = client.messages.create(
      system=[{
          "type": "text",
          "text": "You are a helpful assistant...",
          "cache_control": {"type": "ephemeral"}   ← tell Claude to cache this
      }],
      messages=[{
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": "Context: Employees get 20 days leave...",
                  "cache_control": {"type": "ephemeral"}   ← cache context too
              },
              {
                  "type": "text",
                  "text": "Question: how many leave days?"   ← NOT cached
              }
          ]
      }]
  )

  First call:
    input_tokens:        2000  (full price  $3.00/M)
    cache_write_tokens:  1980  (stored server-side)

  Next 999 calls with same context:
    input_tokens:          20  (just the query)
    cache_read_tokens:   1980  (10% of normal = $0.30/M)

  Cost comparison (1000 queries, 2000 tokens each):
    Without cache:  1000 × 2000 × $3.00/M  = $6.00/day
    With cache:     1 × 2000 × $3.00/M
                  + 999 × 20 × $3.00/M
                  + 999 × 1980 × $0.30/M   = $0.66/day

    Saving: $5.34/day = 89% cheaper ✅
    """)

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
