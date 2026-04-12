"""
Base class for all RAG retrieval strategies.

Every concrete strategy (NaiveRAG, HyDE, SelfRAG, …) inherits from
BaseRAG and overrides `run()`.  The shared state (vector store, embedder,
generator) is held here so subclasses can focus on their unique logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.config import config
from src.documents import RetrievalResult
from src.embeddings.base import BaseEmbedder
from src.generation.generator import Generator, RAGResponse
from src.vectorstore.memory import InMemoryVectorStore


class BaseRAG(ABC):
    """
    Shared infrastructure for all RAG patterns.

    Parameters
    ----------
    store     : InMemoryVectorStore — pre-indexed chunk corpus
    embedder  : BaseEmbedder — fitted on the same corpus
    generator : Generator — wraps the Claude API
    """

    def __init__(
        self,
        store: InMemoryVectorStore,
        embedder: BaseEmbedder,
        generator: Generator,
    ):
        self.store = store
        self.embedder = embedder
        self.generator = generator

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, k: int | None = None) -> List[RetrievalResult]:
        """Embed *query* and search the vector store."""
        k = k or config.top_k
        q_vec = self.embedder.embed_query(query)
        return self.store.search(q_vec, k=k, min_score=0.0)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, query: str) -> RAGResponse:
        """Execute the full RAG pipeline and return a RAGResponse."""
