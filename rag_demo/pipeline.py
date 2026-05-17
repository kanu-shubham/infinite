"""
pipeline.py — The glue that connects every component into one RAG system.

Flow:
    index_document()  →  chunk → embed → store in vector + BM25 + bookkeeper
    query()           →  retrieve → (optional rerank) → generate
"""
from documents import Document, Chunk, chunk_document
from embeddings import TFIDFEmbedder
from vectorstore import VectorStore, RetrievalResult
from bm25 import BM25Index
from hybrid import HybridRetriever
from reranker import CrossEncoderReranker
from bookkeeper import Bookkeeper
from generator import Generator, RAGResponse
from acl import UserContext, can_access


class RAGPipeline:

    def __init__(
        self,
        embedder: TFIDFEmbedder = None,
        use_hybrid: bool = True,
        use_reranker: bool = False,
        db_path: str = ":memory:",
    ):
        self.embedder = embedder or TFIDFEmbedder(n_components=64)
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.bookkeeper = Bookkeeper(db_path=db_path)
        self.generator = Generator()
        self.use_hybrid = use_hybrid
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def _ensure_fitted(self, texts: list[str]) -> None:
        """Fit the TF-IDF embedder if not already fitted."""
        if not self._fitted:
            self.embedder.fit(texts)
            self._fitted = True

    def index_document(
        self,
        doc: Document,
        permissions: set[str] = None,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> list[Chunk]:
        """
        Full indexing pipeline for one document:
        1. Split into chunks
        2. Fit embedder (first time) or use existing fit
        3. Embed all chunks
        4. Store in vector store + BM25 index
        5. Register in bookkeeper
        """
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        self._ensure_fitted(texts)

        embeddings = self.embedder.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            # Attach permission tokens to chunk metadata for post-filter ACL
            if permissions:
                chunk.metadata["permissions"] = permissions

        self.vector_store.add_chunks(chunks)
        self.bm25_index.add_chunks(chunks)
        self.bookkeeper.register_chunks(doc.doc_id, chunks)
        if permissions:
            self.bookkeeper.set_permissions(doc.doc_id, permissions)

        return chunks

    def upsert_document(self, doc: Document, permissions: set[str] = None) -> list[Chunk]:
        """
        Re-index a document that may already exist.
        Old chunks are removed, new chunks are inserted.
        This is idempotent: calling it twice with the same doc produces the same state.
        """
        # Find and delete old chunks
        old_chunk_ids = self.bookkeeper.get_chunk_ids_for_doc(doc.doc_id)
        if old_chunk_ids:
            id_set = set(old_chunk_ids)
            self.vector_store.remove_chunks(id_set)
            self.bm25_index.remove_chunks(id_set)
            self.bookkeeper.delete_document(doc.doc_id)

        return self.index_document(doc, permissions=permissions)

    def delete_document(self, doc_id: str) -> None:
        chunk_ids = set(self.bookkeeper.get_chunk_ids_for_doc(doc_id))
        self.vector_store.remove_chunks(chunk_ids)
        self.bm25_index.remove_chunks(chunk_ids)
        self.bookkeeper.delete_document(doc_id)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def query(
        self,
        query: str,
        user: UserContext = None,
        top_k: int = 5,
        pattern: str = "naive",
    ) -> RAGResponse:
        """
        Retrieve relevant chunks and generate an answer.

        ACL: if a user is provided, filter out chunks they cannot access.
        """
        if not self._fitted:
            raise RuntimeError("Index at least one document before querying.")

        # ── Retrieval ──────────────────────────────────────────────────
        if self.use_hybrid:
            retriever = HybridRetriever(self.bm25_index, self.vector_store, self.embedder)
            results = retriever.search(query, top_k=top_k * 4)
        else:
            q_emb = self.embedder.embed_query(query)
            results = self.vector_store.search(q_emb, top_k=top_k * 4)

        # ── ACL filter ─────────────────────────────────────────────────
        if user is not None:
            results = [
                r for r in results
                if can_access(user, r.chunk.metadata.get("permissions", set()))
            ]

        # ── Optional reranking ─────────────────────────────────────────
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # ── Generation ─────────────────────────────────────────────────
        response = self.generator.generate(query, results, pattern=pattern)

        # ── Audit log ──────────────────────────────────────────────────
        self.bookkeeper.log_query(
            user_id=user.user_id if user else "anonymous",
            query=query,
            num_results=len(results),
            pattern=pattern,
        )

        return response
