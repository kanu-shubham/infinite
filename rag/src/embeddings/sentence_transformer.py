"""
Sentence-Transformers Embedder

Production-quality semantic embeddings using the sentence-transformers
library.  Unlike TF-IDF+SVD, these models:

  - Generalise to words not seen during training (no re-fitting needed)
  - Capture deep semantic meaning (not just co-occurrence)
  - Work out-of-the-box on any domain

Recommended models (speed vs quality tradeoff)
-----------------------------------------------
  all-MiniLM-L6-v2      →  22M params, 384 dims, fastest, good quality
  all-mpnet-base-v2     →  110M params, 768 dims, best quality on CPU
  BAAI/bge-large-en-v1.5 →  335M params, 1024 dims, state-of-the-art

Installation
------------
  pip install sentence-transformers torch

This file is intentionally separate from tfidf.py so that importing
TFIDFEmbedder never requires torch.  Only import this class if you have
sentence-transformers installed.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Wraps a sentence-transformers model as a BaseEmbedder.

    Parameters
    ----------
    model_name : HuggingFace model ID (default: all-MiniLM-L6-v2)
    batch_size : number of texts to encode at once
    device     : "cpu", "cuda", or "mps" (auto-detected if None)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers torch"
            )

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._model_name = model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to L2-normalised float32 vectors.

        sentence-transformers handles batching, GPU transfer, and
        normalisation internally when normalize_embeddings=True.
        """
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,   # L2-normalise so dot == cosine
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def fit(self, texts: List[str]) -> "SentenceTransformerEmbedder":
        """No-op — pre-trained model needs no fitting."""
        return self

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
