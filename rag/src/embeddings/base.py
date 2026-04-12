"""
Abstract embedding interface.

Any embedding back-end must implement `embed_texts`.  This lets you swap
TF-IDF+LSA for sentence-transformers, OpenAI, or Anthropic embeddings
without touching any retrieval code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbedder(ABC):
    """Embed a batch of strings to fixed-size float32 vectors."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray of shape (len(texts), dim), dtype float32.
        Vectors are L2-normalised so that dot-product == cosine similarity.
        """

    def embed_query(self, query: str) -> np.ndarray:
        """Convenience wrapper for a single query string."""
        return self.embed_texts([query])[0]
