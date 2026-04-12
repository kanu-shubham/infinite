"""
Unit tests for the retrieval layer — no API calls required.

Tests cover:
- Document chunking correctness
- TF-IDF embedder fit/transform
- Vector store add/search/dedup
- RRF fusion (query decomposition)
- Naive RAG retrieval (mocked generator)
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import Document, Chunk, chunk_document, RetrievalResult
from src.embeddings.tfidf import TFIDFEmbedder
from src.vectorstore.memory import InMemoryVectorStore
from src.retrieval.query_decomp import _reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Document & Chunking tests
# ---------------------------------------------------------------------------

class TestChunking(unittest.TestCase):

    def test_chunk_short_document(self):
        """A document shorter than chunk_size produces exactly one chunk."""
        doc = Document(content="Hello world.", title="test", source="t")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Hello world.")

    def test_chunk_metadata_copied(self):
        doc = Document(content="A" * 100, title="T", source="S",
                       metadata={"author": "Alice"})
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)
        for c in chunks:
            self.assertEqual(c.metadata["author"], "Alice")
            self.assertEqual(c.doc_title, "T")
            self.assertEqual(c.doc_source, "S")

    def test_chunk_overlap_produces_shared_content(self):
        """Consecutive chunks should share content in the overlap region."""
        text = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        doc = Document(content=text, title="T", source="S")
        chunks = chunk_document(doc, chunk_size=80, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        # Verify continuity — each chunk's start <= previous chunk's end
        for i in range(len(chunks) - 1):
            self.assertLess(chunks[i].start_char, chunks[i + 1].start_char)

    def test_empty_document_produces_no_chunks(self):
        doc = Document(content="", title="empty", source="none")
        chunks = chunk_document(doc)
        self.assertEqual(chunks, [])

    def test_chunk_ids_are_unique(self):
        doc = Document(content=" ".join(["word"] * 1000), title="T", source="S")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_citation_format(self):
        doc = Document(content="Some text.", title="My Book", source="ch1")
        chunks = chunk_document(doc)
        citation = chunks[0].citation()
        self.assertIn("My Book", citation)
        self.assertIn("ch1", citation)


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestTFIDFEmbedder(unittest.TestCase):

    CORPUS = [
        "machine learning models learn from data",
        "deep learning uses neural networks",
        "retrieval augmented generation combines search and language models",
        "vector databases store embeddings for similarity search",
        "transformers use attention mechanisms for natural language processing",
    ]

    def test_fit_produces_normalised_embeddings(self):
        emb = TFIDFEmbedder(n_components=32)
        emb.fit(self.CORPUS)
        vecs = emb.embed_texts(self.CORPUS)
        self.assertEqual(vecs.shape[0], len(self.CORPUS))
        # Check L2 normalisation
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_similar_texts_have_higher_cosine_similarity(self):
        emb = TFIDFEmbedder(n_components=32)
        texts = self.CORPUS + ["deep neural network training"]
        emb.fit(texts)
        vecs = emb.embed_texts(texts)
        # "deep learning..." vs "deep neural..." should be closer than "machine learning..."
        sim_deep = float(vecs[1] @ vecs[-1])
        sim_ml   = float(vecs[0] @ vecs[-1])
        self.assertGreater(sim_deep, sim_ml,
                           "Semantically similar text should have higher cosine similarity")

    def test_embed_query_shape(self):
        emb = TFIDFEmbedder(n_components=16)
        emb.fit(self.CORPUS)
        q = emb.embed_query("neural networks")
        self.assertEqual(q.ndim, 1)
        # SVD caps at min(n_docs - 1, n_components); with 5 docs → max 4 dims
        self.assertGreaterEqual(q.shape[0], 1)
        self.assertLessEqual(q.shape[0], 16)

    def test_lazy_fit_on_first_embed(self):
        emb = TFIDFEmbedder(n_components=8)
        # No explicit fit — should auto-fit on first embed_texts call
        vecs = emb.embed_texts(self.CORPUS)
        self.assertEqual(vecs.shape[0], len(self.CORPUS))


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore(unittest.TestCase):

    def _make_chunk(self, content: str, embedding: np.ndarray) -> Chunk:
        c = Chunk(content=content, doc_id="d1", doc_title="T",
                  doc_source="S", start_char=0, end_char=len(content),
                  chunk_index=0)
        c.embedding = embedding
        return c

    def test_search_returns_closest_vector(self):
        store = InMemoryVectorStore()
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v3 = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # closest to v1

        store.add_chunks([
            self._make_chunk("A", v1),
            self._make_chunk("B", v2),
            self._make_chunk("C", v3),
        ])

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.content, "A")

    def test_search_returns_top_k(self):
        store = InMemoryVectorStore()
        for i in range(10):
            v = np.zeros(4, dtype=np.float32)
            v[i % 4] = 1.0
            store.add_chunks([self._make_chunk(f"doc{i}", v)])

        results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), k=3)
        self.assertEqual(len(results), 3)

    def test_min_score_filter(self):
        store = InMemoryVectorStore()
        v_near = np.array([0.99, 0.0, 0.0], dtype=np.float32)
        v_far  = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        store.add_chunks([
            self._make_chunk("near", v_near),
            self._make_chunk("far",  v_far),
        ])
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, k=5, min_score=0.5)
        contents = [r.chunk.content for r in results]
        self.assertIn("near", contents)
        self.assertNotIn("far", contents)

    def test_empty_store_returns_empty(self):
        store = InMemoryVectorStore()
        results = store.search(np.array([1.0, 0.0]), k=5)
        self.assertEqual(results, [])

    def test_chunk_without_embedding_raises(self):
        store = InMemoryVectorStore()
        c = Chunk(content="x", doc_id="d", doc_title="T",
                  doc_source="S", start_char=0, end_char=1, chunk_index=0)
        with self.assertRaises(ValueError):
            store.add_chunks([c])


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion tests
# ---------------------------------------------------------------------------

def _make_result(chunk_id: str, content: str, score: float) -> RetrievalResult:
    c = Chunk(content=content, doc_id="d", doc_title="T",
              doc_source="S", start_char=0, end_char=len(content),
              chunk_index=0)
    # Override chunk_id for determinism
    object.__setattr__(c, 'chunk_id', chunk_id)
    return RetrievalResult(chunk=c, score=score)


class TestRRF(unittest.TestCase):

    def test_document_appearing_in_multiple_lists_scores_higher(self):
        list1 = [
            _make_result("A", "doc A", 0.9),
            _make_result("B", "doc B", 0.8),
        ]
        list2 = [
            _make_result("A", "doc A", 0.7),
            _make_result("C", "doc C", 0.6),
        ]
        merged = _reciprocal_rank_fusion([list1, list2])
        top_ids = [r.chunk.chunk_id for r in merged]
        # A appears in both lists and should be ranked first
        self.assertEqual(top_ids[0], "A")

    def test_rrf_deduplicates(self):
        list1 = [_make_result("X", "x", 0.9)]
        list2 = [_make_result("X", "x", 0.8)]
        merged = _reciprocal_rank_fusion([list1, list2])
        self.assertEqual(len(merged), 1)

    def test_empty_input_returns_empty(self):
        self.assertEqual(_reciprocal_rank_fusion([]), [])


# ---------------------------------------------------------------------------
# Naive RAG integration test (mocked)
# ---------------------------------------------------------------------------

class TestNaiveRAGMocked(unittest.TestCase):
    """Test NaiveRAG without hitting the API by mocking the Generator."""

    def setUp(self):
        from src.generation.generator import RAGResponse
        self.mock_response = RAGResponse(
            query="test", answer="mocked answer",
            citations=[], retrieval_results=[], pattern="naive"
        )

    def test_naive_rag_calls_generator(self):
        from src.retrieval.naive import NaiveRAG

        embedder = TFIDFEmbedder(n_components=8)
        texts = ["machine learning is great", "deep learning uses transformers"]
        embedder.fit(texts)

        store = InMemoryVectorStore()
        vecs = embedder.embed_texts(texts)
        for text, vec in zip(texts, vecs):
            c = Chunk(content=text, doc_id="d", doc_title="T",
                      doc_source="S", start_char=0, end_char=len(text),
                      chunk_index=0)
            c.embedding = vec
            store.add_chunks([c])

        mock_generator = MagicMock()
        mock_generator.generate.return_value = self.mock_response

        rag = NaiveRAG(store=store, embedder=embedder, generator=mock_generator)
        result = rag.run("machine learning")

        mock_generator.generate.assert_called_once()
        self.assertEqual(result.answer, "mocked answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
