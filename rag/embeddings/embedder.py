"""
embedder.py
===========
Wraps sentence-transformers for generating dense embeddings.

THEORY RECAP (see THEORY.md Section 2 for full theory):
  An embedding model converts text → fixed-size float vector.
  Semantically similar text → close vectors (high cosine similarity).

  We use all-MiniLM-L6-v2 (80MB, 384 dims) as default:
    - Fast to download and run on CPU
    - 56.3 MTEB score — solid baseline
    - Perfect for learning

  Production upgrade: BAAI/bge-large-en-v1.5 (64.2 MTEB, 1024 dims)

KEY CONCEPTS IMPLEMENTED HERE:
  1. Batch embedding  — embed many texts at once (GPU-friendly)
  2. Normalization    — L2-normalize vectors so cosine sim = dot product
  3. Query prefix     — BGE models need "Represent this sentence: " for queries
"""

from typing import List, Union
import numpy as np


class DenseEmbedder:
    """
    Wraps a sentence-transformer model to produce dense embeddings.

    Usage:
        embedder = DenseEmbedder()
        query_vec   = embedder.embed_query("What is a transformer?")
        doc_vecs    = embedder.embed_documents(["Transformers are...", "BERT is..."])

        # Similarity
        sim = np.dot(query_vec, doc_vecs[0])  # cosine sim (vectors are normalized)
    """

    # Best models for different use cases (MTEB scores as of 2025)
    RECOMMENDED_MODELS = {
        "fast":         "all-MiniLM-L6-v2",          # 80MB, 384d, MTEB=56.3
        "balanced":     "BAAI/bge-small-en-v1.5",    # 133MB, 384d, MTEB=62.2
        "best_english": "BAAI/bge-large-en-v1.5",    # 1.3GB, 1024d, MTEB=64.2
        "multilingual": "BAAI/bge-m3",               # 570MB, 1024d, multilingual
        "default":      "all-MiniLM-L6-v2",
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
        use_query_prefix: bool = False,
    ):
        """
        Args:
            model_name:       HuggingFace model id
            batch_size:       texts per batch (larger = faster on GPU, more RAM)
            normalize:        L2-normalize output (required for cosine-as-dot-product)
            use_query_prefix: BGE models perform better with "Represent this question: "
                              prefix on queries. Set True if using BGE models.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.use_query_prefix = use_query_prefix

        print(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

        # Get the embedding dimension from the model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of documents (not queries).

        Returns:
            numpy array of shape (len(texts), embedding_dim)
            Each row is a normalized vector for one document.

        BATCH PROCESSING:
            Instead of embedding one by one, we embed in batches.
            This is ~10x faster because the model processes multiple
            texts in parallel using matrix operations.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 50,  # only show bar for large batches
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        WHY SEPARATE FROM embed_documents?
            Some models (like BGE) are trained with different prefixes for
            queries vs documents:
              Query: "Represent this question for searching relevant passages: {query}"
              Doc:   "{document text}"

            This is called asymmetric embedding. The model learns that queries
            and documents are different types of inputs.

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        text = query
        if self.use_query_prefix:
            text = f"Represent this question for searching relevant passages: {query}"

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.

        If vectors are L2-normalized (normalize=True), this equals dot product.
        Formula: cos(θ) = (a·b) / (|a| × |b|)

        Range: -1 (opposite) to +1 (identical)
        """
        if self.normalize:
            # Optimization: for normalized vectors, cosine sim = dot product
            return float(np.dot(a, b))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def similarity_matrix(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """
        Compute all-pairs cosine similarity between two lists of texts.

        Returns matrix of shape (len(texts_a), len(texts_b))
        Useful for semantic chunking and evaluation.
        """
        vecs_a = self.embed_documents(texts_a)
        vecs_b = self.embed_documents(texts_b)
        # Matrix multiply for fast batch cosine (works because vectors are normalized)
        return vecs_a @ vecs_b.T


if __name__ == "__main__":
    embedder = DenseEmbedder(model_name="all-MiniLM-L6-v2")

    print("\n--- Embedding Dimension ---")
    print(f"Model produces {embedder.embedding_dim}-dimensional vectors")

    print("\n--- Similarity Examples ---")
    texts = [
        "The cat sat on the mat",
        "A kitten rested on a rug",       # semantically similar to text[0]
        "The stock market crashed today",  # semantically different
    ]
    vecs = embedder.embed_documents(texts)
    q_vec = embedder.embed_query("Where did the cat sit?")

    print(f"Query: 'Where did the cat sit?'")
    for i, t in enumerate(texts):
        sim = embedder.cosine_similarity(q_vec, vecs[i])
        print(f"  sim={sim:.4f}  '{t}'")

    print("\n--- Vector Properties ---")
    v = vecs[0]
    print(f"Vector shape: {v.shape}")
    print(f"L2 norm (should be 1.0 since normalize=True): {np.linalg.norm(v):.6f}")
    print(f"First 5 values: {v[:5]}")
