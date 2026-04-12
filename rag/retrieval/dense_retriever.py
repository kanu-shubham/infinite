"""
dense_retriever.py
==================
Dense vector retriever backed by Qdrant (in-memory mode for learning).

THEORY RECAP (see THEORY.md Sections 2 and 4):
  Dense retrieval:
    1. Embed all chunks → store vectors in Qdrant
    2. At query time: embed query → find nearest vectors (cosine similarity)
    3. Return the chunks whose vectors are closest to the query vector

  We use Qdrant in IN-MEMORY mode for this learning project.
  In production: launch Qdrant as a Docker container or use Qdrant Cloud.

QDRANT CONCEPTS:
  Collection = a group of vectors (like a table in a relational DB)
  Point      = one vector + payload (metadata + original text)
  Payload    = the metadata you filter/retrieve (text, source, etc.)
  Vector     = the float array (embedding)

SIMILARITY METRICS IN QDRANT:
  Cosine   = cosine similarity (most common for NLP)
  Dot      = dot product (use when vectors are already normalized → same as cosine)
  Euclid   = L2 Euclidean distance (use for computer vision sometimes)
"""

import uuid
from typing import List, Optional
import numpy as np

from retrieval.bm25_retriever import SearchResult


class DenseRetriever:
    """
    Dense retriever using Qdrant vector database.

    Workflow:
        retriever = DenseRetriever(embedder)
        retriever.index(chunks)           # offline: embed + store
        results = retriever.search(query) # online: embed query + ANN search

    Qdrant in-memory mode:
        Perfect for development and learning.
        No Docker needed, data lives in RAM.
        Switch to production: QdrantClient(url="http://localhost:6333")
    """

    def __init__(self, embedder, collection_name: str = "rag_chunks"):
        """
        Args:
            embedder:         DenseEmbedder instance
            collection_name:  Qdrant collection to store vectors in
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = None
        self._is_indexed = False
        self._id_to_chunk = {}  # map Qdrant point id → Chunk object

    def _init_qdrant(self):
        """Initialize Qdrant in-memory client."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # ":memory:" = in-memory mode, perfect for learning
        # Production: QdrantClient(url="http://localhost:6333")
        self.client = QdrantClient(":memory:")

        # Create collection with cosine similarity
        # Cosine = compare angle between vectors (standard for NLP)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.embedding_dim,
                distance=Distance.COSINE,  # cosine similarity
            ),
        )
        print(f"Qdrant collection '{self.collection_name}' created "
              f"(dims={self.embedder.embedding_dim}, metric=cosine)")

    def index(self, chunks, batch_size: int = 64) -> None:
        """
        Embed all chunks and store in Qdrant.

        INDEXING PIPELINE:
            chunks → batch embed → upsert to Qdrant

        QDRANT POINT STRUCTURE:
            {
              id: UUID (required by Qdrant),
              vector: [0.12, -0.34, ...],  # the embedding
              payload: {                   # your metadata
                chunk_id: "doc_001_rc_000",
                doc_id: "doc_001",
                text: "Neural networks are...",
                title: "Introduction to...",
                ...
              }
            }

        The 'payload' is what you get back when you retrieve a result.
        Store everything you need for the response here.
        """
        from qdrant_client.models import PointStruct

        self._init_qdrant()
        chunks_list = list(chunks)
        print(f"Indexing {len(chunks_list)} chunks into Qdrant...")

        # Process in batches for memory efficiency
        for batch_start in range(0, len(chunks_list), batch_size):
            batch = chunks_list[batch_start:batch_start + batch_size]
            texts = [c.text for c in batch]

            # Embed the batch
            vectors = self.embedder.embed_documents(texts)

            # Create Qdrant points
            points = []
            for chunk, vector in zip(batch, vectors):
                # Qdrant requires UUID-style point IDs
                point_id = str(uuid.uuid4())
                self._id_to_chunk[point_id] = chunk

                points.append(PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "title": chunk.metadata.get("title", ""),
                        "topic": chunk.metadata.get("topic", ""),
                        "chunk_strategy": chunk.metadata.get("chunk_strategy", ""),
                        "parent_id": chunk.parent_id or "",
                        **{k: v for k, v in chunk.metadata.items()
                           if isinstance(v, (str, int, float, bool))},
                    }
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        self._is_indexed = True
        count = self.client.count(self.collection_name).count
        print(f"Indexed {count} vectors in Qdrant.")

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_kwargs: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        Embed query and find the top-k most similar chunks.

        HOW QDRANT SEARCHES:
            1. Embed the query → query_vector
            2. Use HNSW index to find approximate nearest neighbors
            3. Return points with highest cosine similarity to query_vector

        HNSW (Hierarchical Navigable Small World):
            A graph-based ANN algorithm. Builds a multi-layer graph where:
            - Top layers: few nodes, long-range connections (for fast navigation)
            - Bottom layers: dense connections (for precision)
            Achieves >95% recall at millisecond latency for millions of vectors.

        FILTERING:
            Qdrant supports payload filters. Example:
            filter_kwargs = {"must": [{"key": "topic", "match": {"value": "rag"}}]}
            This returns only chunks where topic="rag" AND are semantically similar.

        Returns:
            SearchResult objects ranked by cosine similarity score.
        """
        if not self._is_indexed:
            raise RuntimeError("Call .index() before .search()")

        query_vector = self.embedder.embed_query(query)

        # Build optional filter
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_filter = None
        if filter_kwargs:
            conditions = []
            for key, value in filter_kwargs.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        # Perform ANN search
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        results = []
        for rank, hit in enumerate(hits):
            payload = hit.payload
            results.append(SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                score=float(hit.score),
                rank=rank + 1,
                metadata={k: v for k, v in payload.items() if k != "text"},
            ))

        return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/infinite/rag")
    from corpus.corpus_loader import load_ai_corpus
    from corpus.chunkers import RecursiveCharacterChunker
    from embeddings.embedder import DenseEmbedder

    corpus = load_ai_corpus()
    chunker = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_corpus(corpus)

    embedder = DenseEmbedder()
    retriever = DenseRetriever(embedder)
    retriever.index(chunks)

    queries = [
        "How does the attention mechanism work in transformers?",
        "What are the best embedding models for semantic search?",
        "How to evaluate a RAG pipeline?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  Rank {r.rank} (score={r.score:.4f}): [{r.metadata.get('title', '?')}]")
            print(f"    '{r.text[:100]}...'")
