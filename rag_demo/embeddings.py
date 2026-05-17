"""
embeddings.py — Turn text into vectors (numbers) so we can do similarity search.

We use TF-IDF + Truncated SVD (Latent Semantic Analysis).
In production you'd swap this for a SentenceTransformer model.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class TFIDFEmbedder:
    """
    Two-step pipeline:
      1. TF-IDF: each word → a weight based on how often it appears in this doc
                            vs how rare it is across all docs.
      2. SVD: compress the high-dimensional TF-IDF vector (vocab size ~10k)
              down to `n_components` dimensions (default 128).
              This is also called Latent Semantic Analysis (LSA).

    After SVD the vectors are L2-normalised so cosine_similarity = dot product.
    """

    def __init__(self, n_components: int = 128):
        # TfidfVectorizer converts raw text → sparse TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,   # keep only the 10k most frequent words
            stop_words="english",  # ignore "the", "is", "at", etc.
            ngram_range=(1, 2),    # include single words AND two-word phrases
        )
        # TruncatedSVD is PCA for sparse matrices.
        # It finds the n_components most important "topics" in the corpus.
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.n_components = n_components
        self._fitted = False

    @property
    def dim(self) -> int:
        """The size of each output vector."""
        return self.n_components

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        """
        Learn vocabulary and topic directions from a corpus.
        Must be called before embed_texts/embed_query.
        """
        # Step 1: build vocab and compute TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(texts)  # shape: (N, vocab_size)
        # Step 2: learn the top-n_components topic directions
        self.svd.fit(tfidf_matrix)
        self._fitted = True
        return self

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of texts into a 2D embedding matrix.
        Shape: (len(texts), n_components)
        """
        tfidf = self.vectorizer.transform(texts)       # sparse (N, vocab)
        dense = self.svd.transform(tfidf)              # dense  (N, n_components)
        return normalize(dense, norm="l2")             # each row has unit length

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string → 1D vector of shape (n_components,)
        """
        return self.embed_texts([query])[0]
