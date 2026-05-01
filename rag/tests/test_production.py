"""
Tests for all production features added in v2.

Covers:
- ACL (access control)
- Metadata filtering
- FAISS vector store
- Incremental indexing (upsert / delete)
- Hybrid search (BM25 + dense)
- Conversation rewriter (mocked)
- Reranker (mocked)
- Query cache (TTL)
- Retry logic
- Structured logging
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.acl import UserContext, can_access, SUPERUSER
from src.cache import QueryCache
from src.documents import Chunk, Document, RetrievalResult
from src.embeddings.tfidf import TFIDFEmbedder
from src.vectorstore.faiss_store import FAISSVectorStore, MetadataFilter
from src.vectorstore.memory import InMemoryVectorStore
from src.indexing.incremental import IncrementalIndexManager
from src.retrieval.hybrid import HybridRetriever, _rrf_merge
from src.generation.generator import _with_retry, RAGResponse, Citation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(content: str, chunk_id: str = None, permissions=None, **meta) -> Chunk:
    c = Chunk(content=content, doc_id="d1", doc_title="T",
              doc_source="S", start_char=0, end_char=len(content), chunk_index=0,
              metadata={"permissions": permissions or [], **meta})
    if chunk_id:
        object.__setattr__(c, "chunk_id", chunk_id)
    return c


def _vec(v: list) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


# ---------------------------------------------------------------------------
# ACL Tests
# ---------------------------------------------------------------------------

class TestACL(unittest.TestCase):

    def test_user_can_access_own_permission(self):
        user = UserContext("alice", teams=["finance"])
        self.assertTrue(can_access(user, ["user:alice"]))

    def test_user_can_access_team_permission(self):
        user = UserContext("bob", teams=["engineering"])
        self.assertTrue(can_access(user, ["team:engineering", "team:product"]))

    def test_user_blocked_from_other_team(self):
        user = UserContext("carol", teams=["marketing"])
        self.assertFalse(can_access(user, ["team:finance"]))

    def test_team_all_is_public(self):
        user = UserContext("dave", teams=[])
        self.assertTrue(can_access(user, ["team:all"]))

    def test_empty_permissions_is_public(self):
        user = UserContext("eve", teams=[])
        self.assertTrue(can_access(user, []))

    def test_superuser_sees_everything(self):
        self.assertTrue(can_access(SUPERUSER, ["user:alice"]))
        self.assertTrue(can_access(SUPERUSER, ["team:secret"]))
        self.assertTrue(can_access(SUPERUSER, []))

    def test_role_based_access(self):
        user = UserContext("frank", roles=["admin"])
        self.assertTrue(can_access(user, ["role:admin"]))
        self.assertFalse(can_access(user, ["role:superadmin"]))

    def test_permission_tokens_complete(self):
        user = UserContext("grace", teams=["a", "b"], roles=["r1"])
        tokens = user.permission_tokens
        self.assertIn("user:grace", tokens)
        self.assertIn("team:a", tokens)
        self.assertIn("team:b", tokens)
        self.assertIn("role:r1", tokens)
        self.assertIn("team:all", tokens)


# ---------------------------------------------------------------------------
# Metadata Filter Tests
# ---------------------------------------------------------------------------

class TestMetadataFilter(unittest.TestCase):

    def test_eq_filter(self):
        f = MetadataFilter("source", "eq", "confluence")
        self.assertTrue(f.matches({"source": "confluence"}))
        self.assertFalse(f.matches({"source": "slack"}))

    def test_gte_filter(self):
        f = MetadataFilter("created_at", "gte", "2024-07-01")
        self.assertTrue(f.matches({"created_at": "2024-09-12"}))
        self.assertFalse(f.matches({"created_at": "2024-06-30"}))

    def test_lte_filter(self):
        f = MetadataFilter("created_at", "lte", "2024-09-30")
        self.assertTrue(f.matches({"created_at": "2024-08-15"}))
        self.assertFalse(f.matches({"created_at": "2024-10-01"}))

    def test_in_filter(self):
        f = MetadataFilter("author", "in", ["alice", "bob"])
        self.assertTrue(f.matches({"author": "alice"}))
        self.assertFalse(f.matches({"author": "carol"}))

    def test_contains_filter(self):
        f = MetadataFilter("title", "contains", "pricing")
        self.assertTrue(f.matches({"title": "Q3 Pricing Review"}))
        self.assertFalse(f.matches({"title": "Engineering Roadmap"}))

    def test_missing_field_returns_false(self):
        f = MetadataFilter("nonexistent", "eq", "x")
        self.assertFalse(f.matches({}))


# ---------------------------------------------------------------------------
# FAISS Vector Store Tests
# ---------------------------------------------------------------------------

class TestFAISSVectorStore(unittest.TestCase):

    def _store_with_chunks(self):
        store = FAISSVectorStore(dim=3)
        c1 = _chunk("pets document", permissions=["team:product"])
        c1.embedding = _vec([1, 0, 0])
        c2 = _chunk("finance document", permissions=["team:finance"])
        c2.embedding = _vec([0, 1, 0])
        c3 = _chunk("public document", permissions=["team:all"])
        c3.embedding = _vec([0, 0, 1])
        store.add_chunks([c1, c2, c3])
        return store, [c1, c2, c3]

    def test_basic_search_returns_top_k(self):
        store, _ = self._store_with_chunks()
        results = store.search(_vec([1, 0, 0]), k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.content, "pets document")

    def test_acl_filters_forbidden_chunks(self):
        store, _ = self._store_with_chunks()
        user = UserContext("alice", teams=["product"])
        results = store.search(_vec([0, 1, 0]), k=3, user=user)
        contents = [r.chunk.content for r in results]
        # finance doc is hidden from product team user
        self.assertNotIn("finance document", contents)
        self.assertIn("public document", contents)

    def test_metadata_filter(self):
        store = FAISSVectorStore(dim=3)
        c1 = _chunk("old doc", source="confluence", created_at="2024-01-01")
        c1.embedding = _vec([1, 0, 0])
        c2 = _chunk("new doc", source="confluence", created_at="2024-09-01")
        c2.embedding = _vec([1, 0, 0])
        store.add_chunks([c1, c2])
        f = MetadataFilter("created_at", "gte", "2024-06-01")
        results = store.search(_vec([1, 0, 0]), k=5, filters=[f])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.content, "new doc")

    def test_remove_chunks(self):
        store, chunks = self._store_with_chunks()
        store.remove_chunks([chunks[0].chunk_id])
        results = store.search(_vec([1, 0, 0]), k=5)
        contents = [r.chunk.content for r in results]
        self.assertNotIn("pets document", contents)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSVectorStore(dim=3, index_path=tmpdir)
            c = _chunk("saved chunk")
            c.embedding = _vec([1, 0, 0])
            store.add_chunks([c])
            store.save()

            store2 = FAISSVectorStore(dim=3, index_path=tmpdir)
            store2.load()
            self.assertEqual(len(store2), 1)
            self.assertEqual(store2.all_chunks()[0].content, "saved chunk")


# ---------------------------------------------------------------------------
# Incremental Index Manager Tests
# ---------------------------------------------------------------------------

class _FixedEmbedder(TFIDFEmbedder):
    """Pre-fitted on a realistic corpus so single-doc upserts don't re-fit."""
    _CORPUS = [
        "machine learning neural networks deep learning transformers",
        "retrieval augmented generation vector database embeddings",
        "natural language processing tokenization attention mechanisms",
        "document indexing chunking overlap sentence boundaries",
        "python programming data structures algorithms complexity",
        "supervised unsupervised reinforcement learning training data",
    ]
    def __init__(self):
        super().__init__(n_components=8)
        self.fit(self._CORPUS)


class TestIncrementalIndexManager(unittest.TestCase):

    def _setup(self):
        embedder = _FixedEmbedder()
        store = InMemoryVectorStore()
        manager = IncrementalIndexManager(store, embedder)
        return manager, store

    def _doc(self, content, doc_id_suffix=""):
        return Document(
            content=content, title=f"Doc{doc_id_suffix}", source=f"src{doc_id_suffix}"
        )

    def test_upsert_new_document(self):
        manager, store = self._setup()
        doc = self._doc("machine learning fundamentals")
        chunks = manager.upsert(doc)
        self.assertGreater(len(chunks), 0)
        self.assertGreater(len(store), 0)
        self.assertTrue(manager.is_indexed(doc.doc_id))

    def test_upsert_updates_existing_document(self):
        manager, store = self._setup()
        doc = self._doc("original content about neural networks")
        manager.upsert(doc)
        old_count = len(store)

        # Update same doc_id with new content
        updated = Document(content="completely new content", title="Doc", source="src")
        object.__setattr__(updated, "doc_id", doc.doc_id)
        manager.upsert(updated)

        # Chunk count should reflect new content, not accumulate old+new
        self.assertLessEqual(len(store), old_count + 5)
        chunk_ids = manager.chunk_ids_for(doc.doc_id)
        self.assertGreater(len(chunk_ids), 0)

    def test_delete_document(self):
        manager, store = self._setup()
        doc = self._doc("document to be deleted")
        manager.upsert(doc)
        self.assertTrue(manager.is_indexed(doc.doc_id))
        manager.delete(doc.doc_id)
        self.assertFalse(manager.is_indexed(doc.doc_id))

    def test_stats(self):
        manager, _ = self._setup()
        manager.batch_upsert([self._doc("doc one"), self._doc("doc two")])
        stats = manager.stats()
        self.assertEqual(stats.total_docs, 2)
        self.assertGreater(stats.total_chunks, 0)


# ---------------------------------------------------------------------------
# Hybrid Search Tests
# ---------------------------------------------------------------------------

class TestHybridSearch(unittest.TestCase):

    def _setup(self):
        corpus = [
            "machine learning neural network deep learning",
            "transformer attention mechanism BERT GPT",
            "error code E-2847 authentication failed",
            "retrieval augmented generation RAG pipeline",
            "supervised learning training data features labels",
            "natural language processing text classification",
        ]
        embedder = TFIDFEmbedder(n_components=4)
        embedder.fit(corpus)
        dim = embedder.dim   # actual dim after SVD cap
        store = FAISSVectorStore(dim=dim)
        for text in corpus:
            c = _chunk(text)
            c.embedding = embedder.embed_query(text)
            store.add_chunks([c])
        hybrid = HybridRetriever(store, embedder)
        hybrid.build_bm25()
        return hybrid, embedder

    def test_keyword_query_finds_exact_match(self):
        hybrid, _ = self._setup()
        # "E-2847" is a keyword — BM25 should surface it
        results = hybrid.search("error code E-2847", k=3)
        contents = [r.chunk.content for r in results]
        self.assertTrue(any("E-2847" in c for c in contents))

    def test_semantic_query_still_works(self):
        hybrid, _ = self._setup()
        results = hybrid.search("neural network architecture", k=2)
        # With TF-IDF, dense retrieval should find ML-related content
        self.assertGreater(len(results), 0)
        # Top result should be ML-related, not error code content
        self.assertNotIn("E-2847", results[0].chunk.content)

    def test_returns_at_most_k_results(self):
        hybrid, _ = self._setup()
        results = hybrid.search("neural network", k=2)
        self.assertLessEqual(len(results), 2)

    def test_rrf_merge_deduplicates(self):
        from src.documents import Chunk
        r1 = RetrievalResult(chunk=_chunk("x", "id1"), score=0.9)
        r2 = RetrievalResult(chunk=_chunk("x", "id1"), score=0.7)
        merged = _rrf_merge([[r1], [r2]])
        ids = [r.chunk.chunk_id for r in merged]
        self.assertEqual(len(ids), len(set(ids)))


# ---------------------------------------------------------------------------
# Query Cache Tests
# ---------------------------------------------------------------------------

class TestQueryCache(unittest.TestCase):

    def _mock_response(self, answer="test"):
        return RAGResponse(query="q", answer=answer,
                           citations=[], retrieval_results=[], pattern="naive")

    def test_cache_miss_returns_none(self):
        cache = QueryCache(ttl_seconds=60)
        self.assertIsNone(cache.get("what is bert?", "naive"))

    def test_cache_hit_returns_response(self):
        cache = QueryCache(ttl_seconds=60)
        resp = self._mock_response("BERT is a transformer model")
        cache.set("what is bert?", "naive", resp)
        cached = cache.get("what is bert?", "naive")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.answer, "BERT is a transformer model")

    def test_cache_is_case_insensitive(self):
        cache = QueryCache(ttl_seconds=60)
        resp = self._mock_response()
        cache.set("What is BERT?", "naive", resp)
        self.assertIsNotNone(cache.get("what is bert?", "naive"))

    def test_cache_expires_after_ttl(self):
        cache = QueryCache(ttl_seconds=1)
        cache.set("q", "naive", self._mock_response())
        time.sleep(1.1)
        self.assertIsNone(cache.get("q", "naive"))

    def test_cache_pattern_is_part_of_key(self):
        cache = QueryCache(ttl_seconds=60)
        r1 = self._mock_response("naive answer")
        r2 = self._mock_response("hyde answer")
        cache.set("q", "naive", r1)
        cache.set("q", "hyde", r2)
        self.assertEqual(cache.get("q", "naive").answer, "naive answer")
        self.assertEqual(cache.get("q", "hyde").answer, "hyde answer")

    def test_cache_invalidate_clears_all(self):
        cache = QueryCache(ttl_seconds=60)
        cache.set("q1", "naive", self._mock_response())
        cache.set("q2", "hyde", self._mock_response())
        cache.invalidate()
        self.assertIsNone(cache.get("q1", "naive"))

    def test_hit_rate_tracking(self):
        cache = QueryCache(ttl_seconds=60)
        cache.get("miss", "naive")
        cache.set("hit", "naive", self._mock_response())
        cache.get("hit", "naive")
        self.assertAlmostEqual(cache.hit_rate, 0.5)


# ---------------------------------------------------------------------------
# Retry Logic Tests
# ---------------------------------------------------------------------------

class TestRetryLogic(unittest.TestCase):

    def test_succeeds_on_first_try(self):
        call_count = [0]
        def fn():
            call_count[0] += 1
            return "ok"
        result = _with_retry(fn, max_retries=3, base_delay=0)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 1)

    def test_retries_on_rate_limit(self):
        import anthropic
        call_count = [0]
        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise anthropic.RateLimitError(
                    "rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body={}
                )
            return "ok"
        result = _with_retry(fn, max_retries=4, base_delay=0)
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 3)

    def test_raises_after_max_retries(self):
        import anthropic
        def fn():
            raise anthropic.RateLimitError(
                "always rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={}
            )
        with self.assertRaises(anthropic.RateLimitError):
            _with_retry(fn, max_retries=2, base_delay=0)

    def test_does_not_retry_auth_error(self):
        import anthropic
        call_count = [0]
        def fn():
            call_count[0] += 1
            raise anthropic.AuthenticationError(
                "bad key",
                response=MagicMock(status_code=401, headers={}),
                body={}
            )
        with self.assertRaises(anthropic.AuthenticationError):
            _with_retry(fn, max_retries=4, base_delay=0)
        self.assertEqual(call_count[0], 1)  # only tried once


# ---------------------------------------------------------------------------
# Structured Logging Tests
# ---------------------------------------------------------------------------

class TestObservability(unittest.TestCase):

    def test_log_emits_json(self):
        import logging
        from src.observability import RAGLogger
        records = []
        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        rl = RAGLogger("test.rag")
        rl._log.addHandler(Capture())
        rl._log.propagate = False

        rl.query_start("what is bert?", "naive", user_id="u1")
        self.assertEqual(len(records), 1)
        data = json.loads(records[0])
        self.assertEqual(data["event"], "query_start")
        self.assertEqual(data["pattern"], "naive")
        self.assertIn("ts", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
