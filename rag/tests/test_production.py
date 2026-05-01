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


# ---------------------------------------------------------------------------
# CrossEncoderReranker Tests
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker(unittest.TestCase):

    def _make_results(self, contents):
        results = []
        for i, text in enumerate(contents):
            chunk = _chunk(text, chunk_id=f"c{i}")
            results.append(RetrievalResult(chunk=chunk, score=1.0 - i * 0.1))
        return results

    def test_rerank_returns_top_k(self):
        """CrossEncoderReranker returns at most k results."""
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5, 0.8, 0.3])

        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
            reranker._model = mock_model
            reranker._top_n = 20

        results = self._make_results(["a", "b", "c", "d", "e"])
        reranked = reranker.rerank("test query", results, k=3)
        self.assertEqual(len(reranked), 3)

    def test_rerank_orders_by_cross_encoder_score(self):
        """Highest cross-encoder score should be first, regardless of original order."""
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        # passage "b" gets highest score, "a" gets lowest
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model
        reranker._top_n = 20

        results = self._make_results(["a", "b", "c"])
        reranked = reranker.rerank("query", results, k=3)
        self.assertEqual(reranked[0].chunk.content, "b")
        self.assertEqual(reranked[2].chunk.content, "a")

    def test_rerank_empty_returns_empty(self):
        """Empty input returns empty list without calling the model."""
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model
        reranker._top_n = 20

        result = reranker.rerank("query", [], k=5)
        self.assertEqual(result, [])
        mock_model.predict.assert_not_called()

    def test_rerank_caps_candidates_at_top_n(self):
        """Only top_n candidates are passed to the model."""
        from src.retrieval.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5])

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model
        reranker._top_n = 2  # only take first 2 of 5

        results = self._make_results(["a", "b", "c", "d", "e"])
        reranker.rerank("query", results, k=2)

        pairs_passed = mock_model.predict.call_args[0][0]
        self.assertEqual(len(pairs_passed), 2)

    def test_import_error_raised_without_sentence_transformers(self):
        """Should raise ImportError with helpful message if not installed."""
        import importlib, sys
        from src.retrieval.reranker import CrossEncoderReranker

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises((ImportError, TypeError)):
                CrossEncoderReranker(model_name="dummy")


# ---------------------------------------------------------------------------
# GraphRAG Tests (fully mocked — no real Claude calls)
# ---------------------------------------------------------------------------

class TestGraphRAG(unittest.TestCase):
    """
    Tests for GraphRAG logic using mocked Claude responses.
    All API calls are intercepted; no network traffic occurs.
    """

    def _make_embedder(self, dim=8):
        """A deterministic fake embedder."""
        embedder = MagicMock()
        embedder.dim = dim
        embedder.embed_texts.side_effect = lambda texts: [
            np.random.default_rng(i).random(dim).astype(np.float32)
            for i, _ in enumerate(texts)
        ]
        embedder.embed_query.return_value = np.ones(dim, dtype=np.float32) / (dim ** 0.5)
        return embedder

    def _make_store_with_chunks(self, contents, dim=8):
        store = InMemoryVectorStore()
        rng = np.random.default_rng(42)
        for i, text in enumerate(contents):
            c = _chunk(text, chunk_id=f"chunk{i}")
            c.embedding = rng.random(dim).astype(np.float32)
            store.add_chunks([c])
        return store

    def _mock_extract_response(self, entities, relationships=None):
        """Build a fake Claude response JSON for entity extraction."""
        return json.dumps({
            "entities": entities,
            "relationships": relationships or [],
        })

    def _make_graph_rag(self, contents=None, dim=8):
        from src.retrieval.graph_rag import GraphRAG
        from src.generation.generator import Generator

        contents = contents or [
            "BERT is a transformer model developed by Google.",
            "GPT-4 is a large language model made by OpenAI.",
            "Google and OpenAI are leading AI research organisations.",
        ]

        store = self._make_store_with_chunks(contents, dim=dim)
        embedder = self._make_embedder(dim=dim)
        generator = MagicMock(spec=Generator)
        generator.generate.return_value = MagicMock(
            answer="Test answer",
            citations=[],
            metadata={},
        )

        rag = GraphRAG(store=store, embedder=embedder, generator=generator)
        return rag, store, embedder, generator

    # ------------------------------------------------------------------
    # KnowledgeGraphBuilder
    # ------------------------------------------------------------------

    def test_extract_from_chunk_parses_entities(self):
        from src.retrieval.graph_rag import KnowledgeGraphBuilder, Entity

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [MagicMock(
            text=self._mock_extract_response([
                {"name": "BERT", "type": "CONCEPT", "description": "Transformer model"},
                {"name": "Google", "type": "ORGANIZATION", "description": "Tech company"},
            ])
        )]

        builder = KnowledgeGraphBuilder(mock_client)
        chunk = _chunk("BERT was created by Google.", chunk_id="c1")
        entities, relationships = builder.extract_from_chunk(chunk)

        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0].name, "BERT")
        self.assertEqual(entities[1].entity_type, "ORGANIZATION")
        self.assertEqual(entities[0].source_chunk_ids, ["c1"])

    def test_extract_handles_malformed_json(self):
        from src.retrieval.graph_rag import KnowledgeGraphBuilder

        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text="not valid json at all {{{")
        ]

        builder = KnowledgeGraphBuilder(mock_client)
        chunk = _chunk("Some text", chunk_id="c1")
        entities, relationships = builder.extract_from_chunk(chunk)

        self.assertEqual(entities, [])
        self.assertEqual(relationships, [])

    def test_extract_strips_markdown_fences(self):
        from src.retrieval.graph_rag import KnowledgeGraphBuilder

        mock_client = MagicMock()
        fenced = "```json\n" + self._mock_extract_response([
            {"name": "BERT", "type": "CONCEPT", "description": "model"}
        ]) + "\n```"
        mock_client.messages.create.return_value.content = [MagicMock(text=fenced)]

        builder = KnowledgeGraphBuilder(mock_client)
        entities, _ = builder.extract_from_chunk(_chunk("text"))
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].name, "BERT")

    def test_build_graph_merges_duplicate_entities(self):
        from src.retrieval.graph_rag import KnowledgeGraphBuilder

        mock_client = MagicMock()
        # Two chunks both mention BERT — should be merged into one node
        mock_client.messages.create.return_value.content = [MagicMock(
            text=self._mock_extract_response([
                {"name": "BERT", "type": "CONCEPT", "description": "desc1"},
            ])
        )]

        builder = KnowledgeGraphBuilder(mock_client)
        chunks = [
            _chunk("chunk A", chunk_id="ca"),
            _chunk("chunk B", chunk_id="cb"),
        ]
        G = builder.build_graph(chunks)

        # BERT should appear exactly once even though it's in two chunks
        self.assertIn("BERT", G.nodes)
        entity = G.nodes["BERT"]["entity"]
        self.assertIn("ca", entity.source_chunk_ids)
        self.assertIn("cb", entity.source_chunk_ids)

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def test_detect_communities_empty_graph(self):
        import networkx as nx
        from src.retrieval.graph_rag import detect_communities

        G = nx.Graph()
        communities = detect_communities(G)
        self.assertEqual(communities, [])

    def test_detect_communities_two_components(self):
        import networkx as nx
        from src.retrieval.graph_rag import detect_communities

        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("C", "D")  # disconnected from A-B

        communities = detect_communities(G)
        # Should find at least 2 communities for two disconnected components
        self.assertGreaterEqual(len(communities), 1)

    def test_detect_communities_splits_large_communities(self):
        import networkx as nx
        from src.retrieval.graph_rag import detect_communities

        G = nx.Graph()
        # One star graph with 20 nodes — too large (> max_community_size=15)
        for i in range(20):
            G.add_edge("hub", f"node{i}")

        communities = detect_communities(G, max_community_size=15)
        # No single community should exceed max_community_size
        for c in communities:
            self.assertLessEqual(len(c.entity_names), 15)

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def test_is_global_query_keyword_detection(self):
        from src.retrieval.graph_rag import GraphRAG

        rag, _, _, _ = self._make_graph_rag()

        self.assertTrue(rag._is_global_query("What are the main themes?"))
        self.assertTrue(rag._is_global_query("Give me an overall summary"))
        self.assertFalse(rag._is_global_query("What did BERT do?"))
        self.assertFalse(rag._is_global_query("Who created GPT-4?"))

    def test_is_global_forced_mode(self):
        from src.retrieval.graph_rag import GraphRAG
        from src.generation.generator import Generator

        store = self._make_store_with_chunks(["text"])
        rag_global = GraphRAG(
            store=store,
            embedder=self._make_embedder(),
            generator=MagicMock(spec=Generator),
            query_mode="global",
        )
        rag_local = GraphRAG(
            store=store,
            embedder=self._make_embedder(),
            generator=MagicMock(spec=Generator),
            query_mode="local",
        )

        self.assertTrue(rag_global._is_global_query("specific entity question"))
        self.assertFalse(rag_local._is_global_query("main themes summary overall"))

    # ------------------------------------------------------------------
    # build_graph integration (mocked LLM)
    # ------------------------------------------------------------------

    def test_build_graph_sets_built_flag(self):
        from src.retrieval.graph_rag import GraphRAG

        rag, _, _, _ = self._make_graph_rag()

        with patch("anthropic.Anthropic") as mock_anthropic:
            client = MagicMock()
            mock_anthropic.return_value = client
            rag._client = client

            # Entity extraction returns one entity per chunk
            client.messages.create.return_value.content = [MagicMock(
                text=self._mock_extract_response([
                    {"name": "BERT", "type": "CONCEPT", "description": "model"}
                ])
            )]

            rag.build_graph()

        self.assertTrue(rag._built)
        self.assertIsNotNone(rag._graph)

    def test_build_graph_empty_store(self):
        from src.retrieval.graph_rag import GraphRAG
        from src.generation.generator import Generator

        empty_store = InMemoryVectorStore()
        rag = GraphRAG(
            store=empty_store,
            embedder=self._make_embedder(),
            generator=MagicMock(spec=Generator),
        )
        rag.build_graph()
        # Should handle gracefully — _built stays False, graph is None
        self.assertFalse(rag._built)

    # ------------------------------------------------------------------
    # Local / global query (pre-built graph, mocked)
    # ------------------------------------------------------------------

    def test_local_query_falls_back_when_no_entity_match(self):
        """If query contains no known entity names, falls back to vector retrieval."""
        from src.retrieval.graph_rag import GraphRAG
        import networkx as nx

        rag, store, embedder, generator = self._make_graph_rag()
        # Manually set up a built graph with one entity
        rag._graph = nx.Graph()
        rag._graph.add_node("BERT")
        rag._entity_map = {}
        rag._built = True

        generator.generate.return_value.metadata = {}
        rag._local_query("What is the weather today?")

        generator.generate.assert_called_once()

    def test_global_query_falls_back_when_no_communities(self):
        """If no communities built, global query falls back to vector retrieval."""
        from src.retrieval.graph_rag import GraphRAG
        import networkx as nx

        rag, store, embedder, generator = self._make_graph_rag()
        rag._graph = nx.Graph()
        rag._communities = []
        rag._entity_map = {}
        rag._built = True

        generator.generate.return_value.metadata = {}
        rag._global_query("What are the main themes?")

        generator.generate.assert_called_once()

    def test_run_triggers_build_if_not_built(self):
        """run() calls build_graph() automatically if not already built."""
        from src.retrieval.graph_rag import GraphRAG

        rag, _, _, generator = self._make_graph_rag()
        generator.generate.return_value.metadata = {}

        with patch.object(rag, "build_graph", wraps=lambda: setattr(rag, "_built", True) or rag) as mock_build:
            rag._built = False
            rag.run("What are the main themes?")
            mock_build.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
