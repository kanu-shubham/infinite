"""
TF-IDF + Truncated SVD (Latent Semantic Analysis) embedder.

Why TF-IDF + SVD?
-----------------
* TF-IDF alone gives high-dimensional *sparse* vectors — great for keyword
  search but can't capture semantics (synonym blind).
* Truncated SVD (a.k.a. LSA) projects them into a dense low-dimensional
  space where semantically related words cluster together.
* No GPU required, trivially installable, deterministic.

Production note
---------------
For real deployments replace this with:
  - sentence-transformers (all-MiniLM-L6-v2) for CPU-friendly semantic search
  - OpenAI text-embedding-3-small / text-embedding-3-large
  - Cohere Embed v3
The BaseEmbedder interface remains identical.

All returned vectors are L2-normalised, so cosine_similarity(a, b) == dot(a, b).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    """
    Parameters
    ----------
    n_components : int
        Dimensionality of the LSA embedding space.
        128–256 works well for small-to-medium corpora.
    min_df : int
        Minimum document frequency for vocabulary inclusion.
    """

    def __init__(self, n_components: int = 128, min_df: int = 1):
        self.n_components = n_components
        self.min_df = min_df
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._is_fitted = False
        # Cache of texts seen during fit so we can detect corpus drift
        self._fit_texts: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        """
        Fit the TF-IDF vocabulary and SVD on *texts*.

        Called automatically on first `embed_texts` if not already fitted.
        Call explicitly after indexing the full corpus for best results.
        """
        self._fit_texts = list(texts)
        n_components = min(self.n_components, len(texts) - 1)

        self._vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            sublinear_tf=True,      # log(1+tf) dampens high-freq terms
            ngram_range=(1, 2),     # unigrams + bigrams
            strip_accents="unicode",
            analyzer="word",
            stop_words="english",
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        self._svd = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=10,
            random_state=42,
        )
        self._svd.fit(tfidf_matrix)
        self._is_fitted = True
        return self

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Transform *texts* into L2-normalised LSA vectors.

        If the embedder hasn't been fitted yet, it is fitted on *texts* first
        (useful for quick scripts; for production always call .fit() on the
        full corpus first).
        """
        if not self._is_fitted:
            self.fit(texts)

        tfidf = self._vectorizer.transform(texts)    # sparse (n, vocab)
        dense = self._svd.transform(tfidf)           # dense  (n, n_components)
        return normalize(dense, norm="l2").astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @property
    def dim(self) -> int:
        return self._svd.n_components if self._svd else self.n_components
