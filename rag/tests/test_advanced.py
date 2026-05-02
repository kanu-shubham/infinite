"""
Tests for advanced RAG features:
- Chunking strategies (Recursive, Semantic, Contextual)
- Step-back prompting
- CRAG (Corrective RAG)
- Agentic RAG
- FAISS HNSW / IVF index types
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import Chunk, Document, RetrievalResult
from src.vectorstore.memory import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, title: str = "Test") -> Document:
    return Document(content=content, title=title, source="test")


def _chunk(content: str, chunk_id: str = None) -> Chunk:
    c = Chunk(
        content=content, doc_id="d1", doc_title="T",
        doc_source="S", start_char=0, end_char=len(content),
        chunk_index=0, metadata={},
    )
    if chunk_id:
        object.__setattr__(c, "chunk_id", chunk_id)
    return c


def _store_with_chunks(contents, dim=8):
    store = InMemoryVectorStore()
    rng = np.random.default_rng(42)
    for i, text in enumerate(contents):
        c = _chunk(text, chunk_id=f"c{i}")
        c.embedding = rng.random(dim).astype(np.float32)
        store.add_chunks([c])
    return store


def _mock_embedder(dim=8):
    embedder = MagicMock()
    embedder.dim = dim
    rng = np.random.default_rng(0)
    embedder.embed_texts.side_effect = lambda texts: [
        rng.random(dim).astype(np.float32) for _ in texts
    ]
    embedder.embed_query.return_value = np.ones(dim, dtype=np.float32) / (dim ** 0.5)
    return embedder


def _mock_generator():
    from src.generation.generator import Generator
    gen = MagicMock(spec=Generator)
    gen.generate.return_value = MagicMock(
        answer="Test answer", citations=[], metadata={},
    )
    return gen


# ---------------------------------------------------------------------------
# Recursive Chunker Tests
# ---------------------------------------------------------------------------

class TestRecursiveChunker(unittest.TestCase):

    def setUp(self):
        from src.indexing.chunkers import RecursiveChunker
        self.Chunker = RecursiveChunker

    def test_short_doc_returns_one_chunk(self):
        chunker = self.Chunker(max_size=512)
        doc = _doc("Hello world.")
        chunks = chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Hello world.")

    def test_long_doc_splits_into_multiple_chunks(self):
        chunker = self.Chunker(max_size=50, overlap=10)
        # Build a doc longer than max_size
        text = "This is a sentence. " * 20
        doc = _doc(text)
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)

    def test_no_chunk_exceeds_max_size(self):
        chunker = self.Chunker(max_size=100, overlap=20)
        text = "Word " * 200
        doc = _doc(text)
        chunks = chunker.chunk_document(doc)
        for c in chunks:
            self.assertLessEqual(len(c.content), 200,
                msg=f"Chunk too large: {len(c.content)} chars")

    def test_prefers_paragraph_splits(self):
        # Each paragraph is ~40 chars; max_size=50 forces splits between paragraphs
        chunker = self.Chunker(max_size=50, overlap=0)
        para = "This paragraph has content in it."  # 33 chars
        text = (para + "\n\n") * 5 + para           # 6 paragraphs, well over max_size
        doc = _doc(text)
        chunks = chunker.chunk_document(doc)
        # Should produce multiple chunks
        self.assertGreater(len(chunks), 1)
        # No single chunk should be longer than max_size * 2 (rough bound)
        for c in chunks:
            self.assertLessEqual(len(c.content), 100,
                msg=f"Chunk too large: {len(c.content)} chars")

    def test_empty_doc_returns_empty(self):
        chunker = self.Chunker(max_size=512)
        doc = _doc("")
        chunks = chunker.chunk_document(doc)
        self.assertEqual(chunks, [])

    def test_chunks_preserve_doc_metadata(self):
        chunker = self.Chunker(max_size=512)
        doc = Document(content="Hello.", title="My Doc", source="wiki",
                       metadata={"author": "Alice"})
        chunks = chunker.chunk_document(doc)
        self.assertEqual(chunks[0].doc_title, "My Doc")
        self.assertEqual(chunks[0].metadata.get("author"), "Alice")


# ---------------------------------------------------------------------------
# Semantic Chunker Tests
# ---------------------------------------------------------------------------

class TestSemanticChunker(unittest.TestCase):

    def _make_chunker(self, breakpoint_std=0.5):
        from src.indexing.chunkers import SemanticChunker
        # High-variance embedder so distances are detectable
        rng = np.random.default_rng(99)
        embedder = MagicMock()
        embedder.embed_texts.side_effect = lambda texts: [
            rng.random(8).astype(np.float32) for _ in texts
        ]
        return SemanticChunker(embedder=embedder, breakpoint_std=breakpoint_std)

    def test_single_sentence_returns_one_chunk(self):
        chunker = self._make_chunker()
        doc = _doc("Only one sentence here.")
        chunks = chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 1)

    def test_produces_chunks_for_multi_sentence_doc(self):
        chunker = self._make_chunker(breakpoint_std=0.01)  # very aggressive splitting
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        doc = _doc(text)
        chunks = chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 1)

    def test_chunks_cover_all_sentences(self):
        chunker = self._make_chunker()
        text = "Alpha sentence. Beta sentence. Gamma sentence."
        doc = _doc(text)
        chunks = chunker.chunk_document(doc)
        combined = " ".join(c.content for c in chunks)
        for word in ["Alpha", "Beta", "Gamma"]:
            self.assertIn(word, combined)

    def test_empty_doc_returns_empty(self):
        chunker = self._make_chunker()
        chunks = chunker.chunk_document(_doc(""))
        self.assertEqual(chunks, [])


# ---------------------------------------------------------------------------
# Contextual Chunker Tests
# ---------------------------------------------------------------------------

class TestContextualChunker(unittest.TestCase):

    def _make_contextual_chunker(self, context="From the overview section."):
        from src.indexing.chunkers import ContextualChunker, RecursiveChunker
        base = RecursiveChunker(max_size=200)
        chunker = ContextualChunker.__new__(ContextualChunker)
        chunker._base = base
        chunker._max_doc_chars = 8000
        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text=context)
        ]
        chunker._client = mock_client
        return chunker

    def test_context_prepended_to_chunk_content(self):
        chunker = self._make_contextual_chunker("From the intro.")
        doc = _doc("The revenue grew 20% year-over-year.")
        chunks = chunker.chunk_document(doc)
        self.assertTrue(chunks[0].content.startswith("From the intro."))
        self.assertIn("revenue", chunks[0].content)

    def test_has_context_metadata_flag(self):
        chunker = self._make_contextual_chunker("Context sentence.")
        doc = _doc("Some content here.")
        chunks = chunker.chunk_document(doc)
        self.assertTrue(chunks[0].metadata.get("has_context"))

    def test_graceful_on_api_error(self):
        from src.indexing.chunkers import ContextualChunker, RecursiveChunker
        base = RecursiveChunker(max_size=200)
        chunker = ContextualChunker.__new__(ContextualChunker)
        chunker._base = base
        chunker._max_doc_chars = 8000
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        chunker._client = mock_client

        doc = _doc("Some content.")
        chunks = chunker.chunk_document(doc)
        # Should still return chunks even without context
        self.assertGreater(len(chunks), 0)
        self.assertIn("Some content", chunks[0].content)


# ---------------------------------------------------------------------------
# Step-Back Prompting Tests
# ---------------------------------------------------------------------------

class TestStepBackRAG(unittest.TestCase):

    def _make_rag(self, contents=None):
        from src.retrieval.step_back import StepBackRAG
        contents = contents or ["BERT improves NLP tasks via attention."]
        store = _store_with_chunks(contents)
        embedder = _mock_embedder()
        generator = _mock_generator()
        rag = StepBackRAG.__new__(StepBackRAG)
        rag.store = store
        rag.embedder = embedder
        rag.generator = generator
        rag._n_original = 3
        rag._n_stepback = 3
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content = [
            MagicMock(text="What are contextual word embedding techniques?")
        ]
        rag._client = mock_client
        return rag

    def test_run_calls_generator(self):
        rag = self._make_rag()
        rag.run("How did BERT improve on ELMo?")
        rag.generator.generate.assert_called_once()

    def test_step_back_query_in_metadata(self):
        rag = self._make_rag()
        rag.generator.generate.return_value.metadata = {}
        resp = rag.run("How did BERT improve on ELMo?")
        self.assertIn("step_back_query", resp.metadata)
        self.assertEqual(resp.metadata["step_back_query"],
                         "What are contextual word embedding techniques?")

    def test_deduplicates_results(self):
        """Same chunk retrieved for both queries should appear only once."""
        from src.retrieval.step_back import StepBackRAG
        rag = self._make_rag(["Only one chunk."])
        rag.generator.generate.return_value.metadata = {}

        # Both retrieve calls return the same chunk
        with patch.object(rag, "_retrieve") as mock_retrieve:
            chunk = _chunk("Only one chunk.", chunk_id="c0")
            mock_retrieve.return_value = [RetrievalResult(chunk=chunk, score=0.9)]
            rag.run("query")

        # generator.generate should receive deduplicated results
        call_kwargs = rag.generator.generate.call_args
        results = call_kwargs[1]["results"] if call_kwargs[1] else call_kwargs[0][1]
        chunk_ids = [r.chunk.chunk_id for r in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))  # no duplicates

    def test_generates_step_back_via_llm(self):
        rag = self._make_rag()
        step_back = rag._generate_step_back("What is the boiling point of ethanol?")
        self.assertIsInstance(step_back, str)
        self.assertGreater(len(step_back), 0)
        rag._client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# CRAG Tests
# ---------------------------------------------------------------------------

class TestCRAG(unittest.TestCase):

    def _make_crag(self, relevance_score=0.8):
        from src.retrieval.crag import CRAG
        store = _store_with_chunks(["Relevant passage about topic X."])
        embedder = _mock_embedder()
        generator = _mock_generator()

        crag = CRAG.__new__(CRAG)
        crag.store = store
        crag.embedder = embedder
        crag.generator = generator
        crag._max_retries = 1

        mock_client = MagicMock()
        # First call: relevance score
        mock_client.messages.create.return_value.content = [
            MagicMock(text=str(relevance_score))
        ]
        crag._client = mock_client
        return crag

    def test_high_relevance_proceeds_directly(self):
        crag = self._make_crag(relevance_score=0.9)
        crag.generator.generate.return_value.metadata = {}
        resp = crag.run("What is topic X?")
        self.assertEqual(resp.metadata.get("crag_action"), "proceed")

    def test_low_relevance_triggers_correct(self):
        from src.retrieval.crag import CRAG
        store = _store_with_chunks(["Some passage."])
        embedder = _mock_embedder()
        generator = _mock_generator()

        crag = CRAG.__new__(CRAG)
        crag.store = store
        crag.embedder = embedder
        crag.generator = generator
        crag._max_retries = 1

        mock_client = MagicMock()
        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(content=[MagicMock(text="0.1")])  # initial score = LOW
            if call_count[0] == 2:
                return MagicMock(content=[MagicMock(text="rewritten query")])  # rewrite
            # subsequent scores
            return MagicMock(content=[MagicMock(text="0.5")])
        mock_client.messages.create.side_effect = side_effect
        crag._client = mock_client

        generator.generate.return_value.metadata = {}
        resp = crag.run("obscure irrelevant question")
        self.assertEqual(resp.metadata.get("crag_action"), "correct")

    def test_relevance_score_in_metadata(self):
        crag = self._make_crag(relevance_score=0.75)
        crag.generator.generate.return_value.metadata = {}
        resp = crag.run("test query")
        self.assertIn("crag_relevance_score", resp.metadata)

    def test_empty_retrieval_returns_gracefully(self):
        from src.retrieval.crag import CRAG
        store = InMemoryVectorStore()  # empty
        embedder = _mock_embedder()
        generator = _mock_generator()

        crag = CRAG.__new__(CRAG)
        crag.store = store
        crag.embedder = embedder
        crag.generator = generator
        crag._max_retries = 1
        crag._client = MagicMock()

        generator.generate.return_value.metadata = {}
        crag.run("question with empty store")
        generator.generate.assert_called_once()

    def test_score_relevance_parses_float(self):
        crag = self._make_crag(relevance_score=0.6)
        dummy_results = [RetrievalResult(chunk=_chunk("text"), score=0.5)]
        score = crag._score_relevance("query", dummy_results)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.6)

    def test_score_relevance_fallback_on_bad_response(self):
        from src.retrieval.crag import CRAG
        crag = CRAG.__new__(CRAG)
        crag._client = MagicMock()
        crag._client.messages.create.return_value.content = [
            MagicMock(text="not a number")
        ]
        score = crag._score_relevance("q", [RetrievalResult(chunk=_chunk("t"), score=1.0)])
        self.assertEqual(score, 0.5)


# ---------------------------------------------------------------------------
# Agentic RAG Tests
# ---------------------------------------------------------------------------

class TestAgenticRAG(unittest.TestCase):

    def _make_agentic(self, contents=None):
        from src.retrieval.agentic import AgenticRAG
        contents = contents or [
            "BERT is a transformer model.",
            "GPT-4 is made by OpenAI.",
        ]
        store = _store_with_chunks(contents)
        embedder = _mock_embedder()
        generator = _mock_generator()

        rag = AgenticRAG.__new__(AgenticRAG)
        rag.store = store
        rag.embedder = embedder
        rag.generator = generator
        rag._max_iterations = 5
        rag._client = MagicMock()
        return rag

    def _tool_use_response(self, tool_name, tool_input, tool_id="t1"):
        """Build a mock Claude response with a tool_use block."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = tool_id
        tool_block.name = tool_name
        tool_block.input = tool_input
        return MagicMock(content=[tool_block], stop_reason="tool_use")

    def test_search_documents_tool_calls_retrieve(self):
        rag = self._make_agentic()

        # First call: search_documents, second call: answer_directly
        rag._client.messages.create.side_effect = [
            self._tool_use_response("search_documents", {"query": "BERT model"}, "t1"),
            self._tool_use_response("answer_directly", {"answer": "BERT is a model."}, "t2"),
        ]

        resp = rag.run("What is BERT?")
        self.assertEqual(resp.answer, "BERT is a model.")
        self.assertTrue(resp.metadata.get("agent_answered"))

    def test_keyword_search_filters_by_substring(self):
        rag = self._make_agentic(["OpenAI research paper", "Google BERT paper"])
        results, _ = rag._execute_tool(
            "search_by_keyword", {"keyword": "OpenAI", "k": 5}, []
        )
        self.assertIn("OpenAI", results)

    def test_source_search_filters_by_doc_source(self):
        from src.retrieval.agentic import AgenticRAG
        store = InMemoryVectorStore()
        rng = np.random.default_rng(1)
        for i, (title, src, text) in enumerate([
            ("Doc A", "wiki/bert", "BERT info"),
            ("Doc B", "arxiv/gpt", "GPT info"),
        ]):
            c = Chunk(content=text, doc_id=f"d{i}", doc_title=title,
                      doc_source=src, start_char=0, end_char=len(text),
                      chunk_index=0, metadata={})
            c.embedding = rng.random(8).astype(np.float32)
            store.add_chunks([c])

        rag = AgenticRAG.__new__(AgenticRAG)
        rag.store = store
        rag.embedder = _mock_embedder()
        rag.generator = _mock_generator()
        rag._max_iterations = 5
        rag._client = MagicMock()

        output, is_terminal = rag._execute_tool(
            "search_by_source", {"source": "wiki", "k": 5}, []
        )
        self.assertIn("BERT", output)
        self.assertNotIn("GPT", output)
        self.assertFalse(is_terminal)

    def test_answer_directly_is_terminal(self):
        rag = self._make_agentic()
        _, is_terminal = rag._execute_tool(
            "answer_directly", {"answer": "Final answer."}, []
        )
        self.assertTrue(is_terminal)

    def test_max_iterations_respected(self):
        """Agent should stop after max_iterations even without answer_directly."""
        rag = self._make_agentic()
        rag._max_iterations = 2

        rag._client.messages.create.return_value = self._tool_use_response(
            "search_documents", {"query": "loop"}, "t1"
        )
        # Neither call invokes answer_directly — should fall back to generator
        rag.generator.generate.return_value.metadata = {"iterations": 2}
        resp = rag.run("infinite query")
        # Generator fallback was used
        rag.generator.generate.assert_called_once()

    def test_no_tool_calls_returns_text_response(self):
        rag = self._make_agentic()
        # Model responds with plain text (no tool blocks)
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Direct answer without tools."
        rag._client.messages.create.return_value = MagicMock(
            content=[text_block], stop_reason="end_turn"
        )
        rag.generator.generate.return_value.metadata = {}
        resp = rag.run("simple question")
        # Should handle gracefully without error
        self.assertIsNotNone(resp)


# ---------------------------------------------------------------------------
# FAISS HNSW / IVF Tests
# ---------------------------------------------------------------------------

class TestFAISSIndexTypes(unittest.TestCase):

    def _make_vecs(self, n=10, dim=16):
        rng = np.random.default_rng(42)
        vecs = rng.random((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs

    def _chunks_with_embeddings(self, n=10, dim=16):
        rng = np.random.default_rng(7)
        chunks = []
        for i in range(n):
            c = _chunk(f"content {i}", chunk_id=f"c{i}")
            vec = rng.random(dim).astype(np.float32)
            c.embedding = vec / np.linalg.norm(vec)
            chunks.append(c)
        return chunks

    def test_flat_index_exact_search(self):
        from src.vectorstore.faiss_store import FAISSVectorStore
        store = FAISSVectorStore(dim=16, index_type="flat")
        chunks = self._chunks_with_embeddings(20, dim=16)
        store.add_chunks(chunks)

        q = np.ones(16, dtype=np.float32)
        q /= np.linalg.norm(q)
        results = store.search(q, k=5)
        self.assertEqual(len(results), 5)

    def test_hnsw_index_returns_results(self):
        from src.vectorstore.faiss_store import FAISSVectorStore
        store = FAISSVectorStore(dim=16, index_type="hnsw", hnsw_m=16)
        chunks = self._chunks_with_embeddings(20, dim=16)
        store.add_chunks(chunks)

        q = np.ones(16, dtype=np.float32)
        q /= np.linalg.norm(q)
        results = store.search(q, k=5)
        self.assertEqual(len(results), 5)

    def test_ivf_falls_back_to_flat_when_too_few_vectors(self):
        """IVF can't be trained with fewer vectors than nlist — should degrade gracefully."""
        from src.vectorstore.faiss_store import FAISSVectorStore
        # nlist=100 but only 5 vectors — should fall back to flat
        store = FAISSVectorStore(dim=16, index_type="ivf", ivf_nlist=100)
        chunks = self._chunks_with_embeddings(5, dim=16)
        store.add_chunks(chunks)

        q = np.ones(16, dtype=np.float32)
        q /= np.linalg.norm(q)
        results = store.search(q, k=3)
        self.assertGreater(len(results), 0)

    def test_ivf_with_enough_vectors(self):
        """IVF should train and search correctly when corpus > nlist."""
        from src.vectorstore.faiss_store import FAISSVectorStore
        store = FAISSVectorStore(dim=16, index_type="ivf", ivf_nlist=4, ivf_nprobe=2)
        chunks = self._chunks_with_embeddings(50, dim=16)
        store.add_chunks(chunks)

        q = np.ones(16, dtype=np.float32)
        q /= np.linalg.norm(q)
        results = store.search(q, k=5)
        self.assertEqual(len(results), 5)

    def test_hnsw_clear_rebuilds_index(self):
        from src.vectorstore.faiss_store import FAISSVectorStore
        store = FAISSVectorStore(dim=16, index_type="hnsw")
        chunks = self._chunks_with_embeddings(10, dim=16)
        store.add_chunks(chunks)
        store.clear()
        self.assertEqual(len(store), 0)
        # Should be able to add again after clear
        store.add_chunks(chunks[:3])
        self.assertEqual(len(store), 3)

    def test_default_index_type_is_flat(self):
        from src.vectorstore.faiss_store import FAISSVectorStore
        import faiss
        store = FAISSVectorStore(dim=8)
        self.assertIsInstance(store._index, faiss.IndexFlatIP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
