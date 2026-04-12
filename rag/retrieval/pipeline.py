"""
pipeline.py
===========
The main RAG Pipeline: ties together chunking, indexing, retrieval, and generation.

This is the "production-like" wrapper that a user would call:

    pipeline = RAGPipeline()
    pipeline.build(documents)
    answer, context = pipeline.query("What is self-attention?")
    eval_result = pipeline.evaluate(test_set)

DESIGN NOTES:
    - Separation of concerns: each component is independently testable
    - Lazy loading: models loaded on first use
    - Configurable: swap any component via constructor args
    - Transparent: retrieve() returns chunks so you can inspect what LLM sees
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from corpus.corpus_loader import Document
from corpus.chunkers import RecursiveCharacterChunker, Chunk
from retrieval.bm25_retriever import BM25Retriever, SearchResult
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever, CrossEncoderReranker


@dataclass
class PipelineConfig:
    """
    All configuration for the RAG pipeline in one place.

    TUNING GUIDE:
        chunk_size:         Start at 400-600, decrease if retrieval misses details
        embedding_model:    all-MiniLM-L6-v2 for speed; bge-large for quality
        use_reranker:       Always True in production; False for quick testing
        dense_weight:       1.0 = balanced; increase for semantic queries
        bm25_weight:        1.0 = balanced; increase for keyword queries
        candidates_bm25:    More candidates = better recall, slower
        final_top_k:        How many chunks the LLM sees (3-7 is typical)
    """
    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates_bm25: int = 30
    candidates_dense: int = 30
    fusion_top_k: int = 15
    final_top_k: int = 5

    # Hybrid weights
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    # RRF
    rrf_k: int = 60


@dataclass
class QueryResult:
    """Result of a single RAG query."""
    question: str
    contexts: List[str]
    context_sources: List[str]
    answer: str
    retrieval_time_ms: float
    num_chunks_retrieved: int

    def pretty_print(self):
        print(f"\n{'='*70}")
        print(f"QUESTION: {self.question}")
        print(f"{'='*70}")
        print(f"\nRETRIEVED CONTEXT ({self.num_chunks_retrieved} chunks, "
              f"{self.retrieval_time_ms:.0f}ms):")
        for i, (ctx, src) in enumerate(zip(self.contexts, self.context_sources), 1):
            print(f"\n  [{i}] Source: {src}")
            print(f"      '{ctx[:200]}{'...' if len(ctx) > 200 else ''}'")
        print(f"\nANSWER:")
        print(f"  {self.answer}")
        print(f"{'='*70}")


class SimpleGenerator:
    """
    A simple rule-based answer generator for demo purposes.

    WHY NOT USE AN ACTUAL LLM HERE?
        To keep this demo self-contained and runnable without API keys.
        In production, replace this with any LLM:
          - OpenAI: client.chat.completions.create(...)
          - Anthropic: client.messages.create(...)
          - Ollama: local LLM running on your machine
          - HuggingFace: transformers pipeline

    The interface is simple: generate(question, contexts) → answer string.
    """

    def generate(self, question: str, contexts: List[str]) -> str:
        """
        Generate an answer using retrieved context.

        In a real system, you'd call an LLM with a prompt like:

            You are a helpful assistant. Answer the question using ONLY
            the information in the provided context.

            Context:
            {context_1}
            {context_2}
            ...

            Question: {question}
            Answer:

        Here we just summarize the context for demonstration.
        """
        if not contexts:
            return "I couldn't find relevant information to answer this question."

        # Find the most relevant sentences (simple keyword matching)
        question_words = set(question.lower().split())
        best_sentences = []

        for ctx in contexts[:3]:  # use top 3 contexts
            sentences = ctx.split('. ')
            for sent in sentences:
                # Score sentence by question word overlap
                sent_words = set(sent.lower().split())
                overlap = len(question_words & sent_words)
                if overlap >= 2:
                    best_sentences.append((overlap, sent.strip()))

        best_sentences.sort(reverse=True)

        if best_sentences:
            # Take top 3 unique sentences
            seen = set()
            answer_parts = []
            for _, sent in best_sentences[:5]:
                if sent not in seen and len(sent) > 30:
                    seen.add(sent)
                    answer_parts.append(sent)
                if len(answer_parts) >= 3:
                    break

            if answer_parts:
                return " ".join(answer_parts) + "."

        # Fallback: first sentence of best context
        first_sentence = contexts[0].split('.')[0]
        return first_sentence + " [See context for full details]"

    def generate_with_llm_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Returns a ready-to-use prompt for an actual LLM.
        Use this to plug in OpenAI, Anthropic, or any other LLM.
        """
        context_text = "\n\n".join(
            f"[Context {i+1}]:\n{ctx}"
            for i, ctx in enumerate(contexts)
        )
        return f"""You are a helpful assistant. Answer the question using ONLY the information provided in the contexts below. If the answer is not in the contexts, say "I don't have enough information to answer this."

{context_text}

Question: {question}
Answer:"""


class RAGPipeline:
    """
    Full RAG pipeline: documents → chunks → index → query → answer.

    Usage:
        # One-time setup
        pipeline = RAGPipeline()
        pipeline.build(documents)

        # Per-query
        result = pipeline.query("What is self-attention?")
        result.pretty_print()

        # Evaluation
        test_cases = [
            {"question": "...", "ground_truth": "..."},
            ...
        ]
        eval_results = pipeline.evaluate_no_llm(test_cases)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.chunker = None
        self.embedder = None
        self.bm25_retriever = None
        self.dense_retriever = None
        self.hybrid_retriever = None
        self.generator = SimpleGenerator()
        self._is_built = False
        self._parent_store: Dict[str, Chunk] = {}  # for parent-doc retrieval

    def _load_embedder(self, cfg: "PipelineConfig", corpus_texts: List[str]):
        """
        Load best available embedder:
          1. Neural (sentence-transformers) — best quality, needs network/cache
          2. TF-IDF + SVD                   — offline fallback, no network needed

        Both share the same API: .embed_query(), .embed_documents(), .embedding_dim
        so the rest of the pipeline is completely unaware of which one is used.
        """
        try:
            from embeddings.embedder import DenseEmbedder
            embedder = DenseEmbedder(model_name=cfg.embedding_model, normalize=True)
            return embedder
        except Exception as e:
            print(f"  Neural model unavailable ({type(e).__name__}). "
                  f"Using offline TF-IDF+SVD embedder.")
            from embeddings.local_embedder import LocalTFIDFEmbedder
            embedder = LocalTFIDFEmbedder(n_components=128)
            embedder.fit(corpus_texts)
            return embedder

    def build(self, documents: List[Document], verbose: bool = True) -> None:
        """
        Index documents: chunk → embed → store.

        This is the offline step that you run once.
        After this, .query() is fast.

        Steps:
            1. Chunk documents
            2. Load embedding model
            3. Build BM25 index (fast)
            4. Build dense index in Qdrant (slower: embedding all chunks)
            5. Load reranker model (if enabled)
        """
        cfg = self.config
        if verbose:
            print(f"\n{'='*60}")
            print(f"Building RAG Pipeline")
            print(f"{'='*60}")
            print(f"Config: chunk_size={cfg.chunk_size}, "
                  f"model={cfg.embedding_model}, reranker={cfg.use_reranker}")

        # Step 1: Chunk
        t0 = time.time()
        self.chunker = RecursiveCharacterChunker(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        chunks = self.chunker.chunk_corpus(documents)
        if verbose:
            print(f"\n[1/4] Chunked {len(documents)} docs → {len(chunks)} chunks "
                  f"({time.time()-t0:.1f}s)")

        # Step 2: Embedding model
        # Try neural (sentence-transformers) first; fall back to TF-IDF+SVD offline.
        t0 = time.time()
        self.embedder = self._load_embedder(cfg, [c.text for c in chunks])
        if verbose:
            model_name = getattr(self.embedder, 'model_name', type(self.embedder).__name__)
            print(f"[2/4] Embedder ready: {model_name} "
                  f"(dim={self.embedder.embedding_dim}, {time.time()-t0:.1f}s)")

        # Step 3: BM25 index
        t0 = time.time()
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.index(chunks)
        if verbose:
            print(f"[3/4] BM25 indexed ({time.time()-t0:.1f}s)")

        # Step 4: Dense index (Qdrant)
        t0 = time.time()
        self.dense_retriever = DenseRetriever(self.embedder)
        self.dense_retriever.index(chunks)
        if verbose:
            print(f"[4/4] Dense index built in Qdrant ({time.time()-t0:.1f}s)")

        # Step 5: Reranker (optional) — needs network/cache for model download
        reranker = None
        if cfg.use_reranker:
            t0 = time.time()
            try:
                reranker = CrossEncoderReranker(model_name=cfg.reranker_model)
                if verbose:
                    print(f"[+] Cross-encoder reranker loaded ({time.time()-t0:.1f}s)")
            except Exception as e:
                if verbose:
                    print(f"[!] Reranker unavailable ({type(e).__name__}). "
                          f"Proceeding with RRF fusion only.")

        # Assemble hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            reranker=reranker,
            bm25_top_k=cfg.candidates_bm25,
            dense_top_k=cfg.candidates_dense,
            fusion_top_k=cfg.fusion_top_k,
            final_top_k=cfg.final_top_k,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_weight=cfg.bm25_weight,
        )

        self._is_built = True
        if verbose:
            print(f"\nPipeline ready!")

    def retrieve(self, question: str, verbose: bool = False) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a question (without generating an answer).
        Use this to inspect what the LLM would see.
        """
        if not self._is_built:
            raise RuntimeError("Call .build(documents) first")
        return self.hybrid_retriever.search(question, verbose=verbose)

    def query(self, question: str, verbose: bool = False) -> QueryResult:
        """
        Full RAG: retrieve relevant chunks + generate answer.

        Returns a QueryResult with the answer AND the contexts used.
        Inspect contexts to understand WHY the answer is what it is.
        """
        if not self._is_built:
            raise RuntimeError("Call .build(documents) first")

        t0 = time.time()
        results = self.retrieve(question, verbose=verbose)
        retrieval_ms = (time.time() - t0) * 1000

        contexts = [r.text for r in results]
        sources = [r.metadata.get("title", r.doc_id) for r in results]

        answer = self.generator.generate(question, contexts)

        return QueryResult(
            question=question,
            contexts=contexts,
            context_sources=sources,
            answer=answer,
            retrieval_time_ms=retrieval_ms,
            num_chunks_retrieved=len(results),
        )

    def get_llm_prompt(self, question: str) -> str:
        """
        Get the full prompt that would be sent to an LLM.
        Use this to plug into OpenAI/Anthropic/Ollama.
        """
        results = self.retrieve(question)
        contexts = [r.text for r in results]
        return self.generator.generate_with_llm_prompt(question, contexts)

    def evaluate_no_llm(self, test_cases: List[Dict]) -> Dict:
        """
        Lightweight evaluation without requiring an LLM judge.
        Uses keyword overlap as a proxy metric.

        For full RAGAS evaluation (LLM-as-judge), see evaluation/ragas_eval.py

        Args:
            test_cases: list of {"question": ..., "ground_truth": ...}

        Returns:
            Dict with per-case results and aggregate metrics.
        """
        results = []
        for case in test_cases:
            question = case["question"]
            ground_truth = case.get("ground_truth", "")

            retrieved = self.retrieve(question)
            contexts = [r.text for r in retrieved]
            answer = self.generator.generate(question, contexts)

            # Keyword-based context recall (proxy for real recall)
            gt_words = set(ground_truth.lower().split()) - {"the", "a", "an", "is", "are", "and"}
            context_text = " ".join(contexts).lower()
            context_words = set(context_text.split())
            recall = len(gt_words & context_words) / len(gt_words) if gt_words else 0

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "context_keyword_recall": round(recall, 3),
                "num_chunks": len(retrieved),
                "top_source": retrieved[0].metadata.get("title", "") if retrieved else "",
            })

        avg_recall = sum(r["context_keyword_recall"] for r in results) / len(results) if results else 0
        return {
            "num_questions": len(results),
            "avg_context_keyword_recall": round(avg_recall, 3),
            "per_question": results,
        }
