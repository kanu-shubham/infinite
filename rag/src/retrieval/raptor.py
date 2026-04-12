"""
RAPTOR — Recursive Abstractive Processing Tree for Organized Retrieval
(Sarthi et al., 2024 — https://arxiv.org/abs/2401.18059)

Core Idea
---------
Naive RAG only retrieves short, local chunks.  RAPTOR builds a *tree*:

  Level 0: original leaf chunks  (fine-grained)
  Level 1: summaries of clusters of leaf chunks
  Level 2: summaries of summaries
  …

At query time, retrieval spans ALL levels of the tree.  This means:
- Specific factual questions are answered by leaf-level chunks.
- High-level conceptual questions are answered by summary nodes.

Building the Tree
-----------------
  1. Embed all leaf chunks.
  2. Cluster embeddings (k-means or agglomerative).
  3. For each cluster, ask the LLM to write a summary.
  4. Embed the summaries → they become the next level's "chunks".
  5. Repeat until only one cluster (or max_levels reached).

Retrieval
---------
  query → embed → search ALL levels simultaneously → rank by score → top-k

When does RAPTOR help?
----------------------
- Long documents where relevant info spans many chunks.
- Thematic / conceptual questions ("What is the overall approach to…?")
- Multi-document corpora needing cross-doc synthesis.

When is it overkill?
--------------------
- Small corpora (< 50 chunks) — overhead is not worth it.
- Simple factual lookups — leaf-level naive RAG is sufficient.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import anthropic
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.config import config
from src.documents import Chunk, RetrievalResult
from src.embeddings.base import BaseEmbedder
from src.generation.generator import Generator, RAGResponse
from src.vectorstore.memory import InMemoryVectorStore
from .base import BaseRAG


_SUMMARISE_SYSTEM = textwrap.dedent("""\
    You are a precise technical summariser.
    Write a dense, factual summary of the following passages in 3–5 sentences.
    Preserve key entities, numbers, and claims.
    Do not add information not present in the passages.
""")


@dataclass
class TreeNode:
    """One node in the RAPTOR summary tree."""
    content: str
    level: int          # 0 = leaf, 1+ = summary
    children: List["TreeNode"] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # Back-reference to the original Chunk (only set for level-0 nodes)
    source_chunk: Optional[Chunk] = None

    def to_chunk(self, doc_id: str = "raptor") -> Chunk:
        """Convert this node to a Chunk for storage in the vector store."""
        return Chunk(
            content=self.content,
            doc_id=doc_id,
            doc_title=f"RAPTOR-L{self.level}",
            doc_source=f"raptor_level_{self.level}",
            start_char=0,
            end_char=len(self.content),
            chunk_index=0,
            metadata={"raptor_level": self.level},
        )


class RaptorIndex:
    """
    Builds and queries the RAPTOR tree.

    Usage
    -----
    idx = RaptorIndex(embedder, client)
    idx.build(leaf_chunks)          # build tree once at index time
    results = idx.search(query, k=5)
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        client: anthropic.Anthropic,
        cluster_size: int = 5,
        max_levels: int = 3,
    ):
        self.embedder = embedder
        self.client = client
        self.cluster_size = cluster_size
        self.max_levels = max_levels
        # Flat store across all tree levels
        self._all_store = InMemoryVectorStore()

    def build(self, leaf_chunks: List[Chunk]) -> None:
        """
        Build the RAPTOR tree from *leaf_chunks* and index every level.
        """
        if not leaf_chunks:
            return

        # Embed leaves
        texts = [c.content for c in leaf_chunks]
        embeddings = self.embedder.embed_texts(texts)
        nodes: List[TreeNode] = []
        for chunk, emb in zip(leaf_chunks, embeddings):
            chunk.embedding = emb
            node = TreeNode(content=chunk.content, level=0,
                            embedding=emb, source_chunk=chunk)
            nodes.append(node)

        # Add leaves to the flat store
        for node in nodes:
            c = node.to_chunk()
            c.embedding = node.embedding
            self._all_store.add_chunks([c])

        # Build summary levels
        current_nodes = nodes
        for level in range(1, self.max_levels + 1):
            if len(current_nodes) < 2:
                break
            summary_nodes = self._build_level(current_nodes, level)
            if not summary_nodes:
                break
            for sn in summary_nodes:
                c = sn.to_chunk()
                c.embedding = sn.embedding
                self._all_store.add_chunks([c])
            current_nodes = summary_nodes

    def _build_level(
        self, nodes: List[TreeNode], level: int
    ) -> List[TreeNode]:
        """Cluster *nodes*, summarise each cluster → next-level nodes."""
        n_clusters = max(1, len(nodes) // self.cluster_size)
        embeddings = np.stack([n.embedding for n in nodes])

        if n_clusters == 1:
            clusters = {0: nodes}
        else:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)
            clusters: dict[int, List[TreeNode]] = {}
            for node, label in zip(nodes, labels):
                clusters.setdefault(int(label), []).append(node)

        summary_nodes: List[TreeNode] = []
        for cluster_nodes in clusters.values():
            passages = "\n\n".join(f"Passage: {n.content}" for n in cluster_nodes)
            summary = self._summarise(passages)
            emb = self.embedder.embed_query(summary)
            sn = TreeNode(
                content=summary,
                level=level,
                children=cluster_nodes,
                embedding=emb,
            )
            summary_nodes.append(sn)

        return summary_nodes

    def _summarise(self, passages: str) -> str:
        resp = self.client.messages.create(
            model=config.model,
            max_tokens=256,
            temperature=0,
            system=_SUMMARISE_SYSTEM,
            messages=[{"role": "user", "content": passages[:4000]}],
        )
        return resp.content[0].text.strip()

    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Search across all tree levels simultaneously."""
        q_vec = self.embedder.embed_query(query)
        return self._all_store.search(q_vec, k=k, min_score=0.0)


# ---------------------------------------------------------------------------
# RAG Strategy
# ---------------------------------------------------------------------------

class RaptorRAG(BaseRAG):
    """
    RAPTOR: retrieve from a hierarchical summary tree.

    The tree is built once during initialisation (or lazily on first run).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._raptor_index: Optional[RaptorIndex] = None

    def build_tree(self) -> None:
        """
        Build the RAPTOR tree from all chunks in the vector store.
        Call this once after indexing documents.
        """
        leaf_chunks = self.store.all_chunks()
        self._raptor_index = RaptorIndex(
            embedder=self.embedder,
            client=self._client,
            cluster_size=config.raptor_cluster_size,
            max_levels=config.raptor_max_levels,
        )
        self._raptor_index.build(leaf_chunks)

    def run(self, query: str) -> RAGResponse:
        if self._raptor_index is None:
            # Lazy build — prefer calling build_tree() explicitly
            self.build_tree()

        results = self._raptor_index.search(query, k=config.top_k)

        response = self.generator.generate(
            query=query,
            results=results,
            pattern="raptor",
        )
        return response
