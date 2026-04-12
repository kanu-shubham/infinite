"""
local_embedder.py
=================
Offline embedding using TF-IDF + SVD (Latent Semantic Analysis).

WHY THIS EXISTS:
    Neural embedding models (sentence-transformers) need network access to
    download from HuggingFace. In offline/restricted environments we use
    TF-IDF + SVD as a stand-in.

    This is NOT just a dummy — TF-IDF + Truncated SVD = Latent Semantic Analysis (LSA),
    a real embedding technique used in production before neural models.

    THEORY: TF-IDF + SVD
    ─────────────────────
    1. TF-IDF builds a sparse bag-of-words matrix:
       Each document → vector of TF-IDF weights for every term in vocabulary
       Shape: (num_docs, vocab_size) — very high dimensional, very sparse

    2. Truncated SVD (= LSA) reduces dimensionality:
       A_sparse → U × Σ × V^T  (Singular Value Decomposition)
       Keep only top-d singular vectors → dense (num_docs, d) matrix

    The resulting dense vectors capture:
    - Semantic similarity between documents (LSA "discovers" synonyms)
    - Topic structure (similar topics cluster together)
    - Far less memory than sparse TF-IDF

    LIMITATION vs neural embeddings:
    - Can't understand long-range context (just bag-of-words)
    - Vocabulary must be known at fit time
    - No multilingual support without translation
    - MTEB scores ~40-45 vs ~56-64 for neural models

    FOR LEARNING: the pipeline architecture is identical — you can
    swap this for SentenceTransformer when network is available.
"""

import numpy as np
from typing import List


class LocalTFIDFEmbedder:
    """
    TF-IDF + SVD embedder that works fully offline.

    Compatible API with DenseEmbedder — drop-in replacement.

    Usage:
        embedder = LocalTFIDFEmbedder(n_components=128)
        embedder.fit(all_texts)  # trains TF-IDF + SVD
        query_vec = embedder.embed_query("self-attention transformer")
        doc_vecs  = embedder.embed_documents(["BERT is a transformer", ...])
    """

    def __init__(self, n_components: int = 128, max_features: int = 10000):
        """
        Args:
            n_components:  Embedding dimension (like 384 for MiniLM)
                          Fewer = faster but less expressive
            max_features:  Max vocabulary size for TF-IDF
        """
        self.n_components = n_components
        self.max_features = max_features
        self.embedding_dim = n_components
        self.vectorizer = None
        self.svd = None
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "LocalTFIDFEmbedder":
        """
        Train TF-IDF vocabulary and SVD projection on a corpus.

        This is the "offline training" step (analogous to downloading a
        pre-trained neural model — but we train it ourselves here).

        Steps:
            1. TfidfVectorizer: learn vocabulary + IDF weights from all texts
            2. TruncatedSVD: learn the top-d semantic directions

        Args:
            texts: list of strings to fit on (typically your whole document corpus)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline

        print(f"Fitting TF-IDF + SVD on {len(texts)} texts...")

        # TF-IDF: term frequency × inverse document frequency
        # sublinear_tf=True → log(1+tf) instead of raw tf (reduces impact of very common terms)
        # min_df=1 → include terms appearing in at least 1 doc
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            sublinear_tf=True,
            min_df=1,
            ngram_range=(1, 2),   # unigrams + bigrams for better phrase matching
            analyzer='word',
            stop_words='english', # remove stop words (the, a, is, ...)
        )

        # Truncated SVD for dimensionality reduction (LSA)
        # Note: PCA doesn't work on sparse matrices; TruncatedSVD does
        self.svd = TruncatedSVD(
            n_components=min(self.n_components, len(texts) - 1),
            algorithm='randomized',
            n_iter=5,
            random_state=42,
        )

        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf_matrix)

        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"TF-IDF vocab: {len(self.vectorizer.vocabulary_)} terms")
        print(f"SVD {self.svd.n_components}d explains {explained_var:.1%} of variance")

        self.embedding_dim = self.svd.n_components
        self._is_fitted = True
        return self

    def _transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to L2-normalized SVD vectors."""
        if not self._is_fitted:
            raise RuntimeError("Call .fit(texts) first")

        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)

        # L2 normalize (makes cosine similarity = dot product)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vecs / norms

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of document strings. Returns (N, dim) array."""
        if not texts:
            return np.array([])
        return self._transform(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns 1D array of shape (dim,)."""
        vecs = self._transform([query])
        return vecs[0]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (= dot product for normalized vectors)."""
        return float(np.dot(a, b))


def get_embedder(texts_for_fitting: List[str] = None, n_components: int = 128):
    """
    Factory: returns a ready-to-use embedder.

    Tries neural (sentence-transformers) first.
    Falls back to TF-IDF + SVD if network/models unavailable.

    Args:
        texts_for_fitting: needed only for TF-IDF fallback
        n_components:      embedding dimensions for TF-IDF fallback
    """
    try:
        # Try neural embedder first
        from sentence_transformers import SentenceTransformer
        import os
        # Attempt a quick load (will fail if no network and not cached)
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        print(f"Using neural embedder: {model_name}")
        from embeddings.embedder import DenseEmbedder
        return DenseEmbedder(model_name=model_name)

    except Exception as e:
        print(f"Neural model unavailable ({type(e).__name__}). Using TF-IDF + SVD offline embedder.")
        if texts_for_fitting is None:
            raise RuntimeError("Provide texts_for_fitting for TF-IDF fallback")
        embedder = LocalTFIDFEmbedder(n_components=n_components)
        embedder.fit(texts_for_fitting)
        return embedder


if __name__ == "__main__":
    texts = [
        "The cat sat on the mat",
        "A kitten rested on a rug",
        "The stock market crashed",
        "Interest rates rose sharply",
        "Neural networks learn from data",
        "Deep learning requires GPUs",
    ]

    embedder = LocalTFIDFEmbedder(n_components=16)
    embedder.fit(texts)

    query = "Where did the cat rest?"
    q_vec = embedder.embed_query(query)
    doc_vecs = embedder.embed_documents(texts)

    print(f"\nQuery: '{query}'")
    print(f"Vector shape: {q_vec.shape}")
    print(f"\nCosine similarities:")
    for i, (text, vec) in enumerate(zip(texts, doc_vecs)):
        sim = embedder.cosine_similarity(q_vec, vec)
        print(f"  {sim:.4f}  '{text}'")
