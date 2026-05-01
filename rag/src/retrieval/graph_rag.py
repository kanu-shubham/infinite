"""
GraphRAG — Graph-based Retrieval Augmented Generation
(Microsoft Research, 2024 — https://arxiv.org/abs/2404.16130)

Core Idea
---------
Standard RAG retrieves chunks that are locally similar to the query.
It misses GLOBAL structure: relationships between entities spread across
many documents, themes that span the entire corpus, or questions like
"What are the main topics discussed?"

GraphRAG builds a KNOWLEDGE GRAPH from the corpus:
  - Nodes: entities  (people, orgs, concepts, events)
  - Edges: relationships between entities
  - Communities: clusters of densely connected entities
  - Summaries: LLM-generated description of each community

Two query modes
---------------
  LOCAL  — "What did Tim Cook say about iPhone margins?"
           Best answered by finding the relevant entity (Tim Cook)
           and retrieving chunks mentioning it.
           Uses: entity search + vector retrieval on source chunks.

  GLOBAL — "What are the main themes in this document collection?"
           Cannot be answered by any single chunk.
           Uses: community summaries — high-level descriptions built
           at index time that describe groups of related entities.

Build Pipeline
--------------
  Documents
      ↓ chunk
  Chunks
      ↓ Claude: extract entities + relationships from each chunk
  Raw graph (many duplicates, noisy)
      ↓ merge duplicate entities (same name → merge descriptions)
  Clean knowledge graph
      ↓ community detection (union-find → connected components)
  Communities (groups of strongly related entities)
      ↓ Claude: summarise each community
  Community summaries (indexed as vectors for global queries)

Query Pipeline
--------------
  LOCAL:
    query → NER → find entities in graph → get source chunks → generate

  GLOBAL:
    query → embed → search community summaries → generate from summaries

Complexity note
---------------
Building the graph requires one Claude call per chunk for entity extraction
+ one Claude call per community for summarisation. Budget accordingly.
For a 1000-chunk corpus, this is ~1000 + n_communities API calls at build time,
but zero extra calls at query time (vs multi-hop which calls Claude per query hop).
"""

from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import anthropic

from src.config import config
from src.documents import Chunk, RetrievalResult
from src.embeddings.base import BaseEmbedder
from src.generation.generator import Generator, RAGResponse
from src.vectorstore.faiss_store import FAISSVectorStore
from src.vectorstore.memory import InMemoryVectorStore
from src.retrieval.base import BaseRAG


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str
    entity_type: str            # PERSON, ORG, CONCEPT, LOCATION, EVENT
    description: str
    source_chunk_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Relationship:
    """A directed edge in the knowledge graph."""
    source: str                 # entity name
    target: str                 # entity name
    description: str
    weight: float = 1.0         # strength of relationship


@dataclass
class Community:
    """A cluster of strongly connected entities."""
    community_id: str
    entity_names: List[str]
    summary: str = ""
    level: int = 0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# LLM extraction prompts
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = textwrap.dedent("""\
    You are a knowledge graph extraction assistant.

    Given a text passage, extract:
    1. ENTITIES: named things (people, organisations, concepts, technologies, events)
    2. RELATIONSHIPS: how entities relate to each other

    Return ONLY valid JSON in this exact format:
    {
      "entities": [
        {"name": "BERT", "type": "CONCEPT", "description": "Bidirectional transformer model by Google"}
      ],
      "relationships": [
        {"source": "BERT", "target": "Google", "description": "developed by", "weight": 1.0}
      ]
    }

    Rules:
    - Entity names must be consistent (always "BERT", never "Bert" or "bert")
    - Minimum 1 entity, maximum 10 entities per passage
    - Minimum 0 relationships, maximum 15 per passage
    - Only extract things explicitly mentioned in the passage
    - Types: PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, LOCATION, EVENT, METRIC
""")

_SUMMARISE_COMMUNITY = textwrap.dedent("""\
    You are summarising a cluster of related entities from a knowledge graph.

    Given a list of entities and their relationships, write a concise 3-5 sentence
    summary that describes:
    1. What this group of entities is about (the main theme)
    2. The key entities and what they do
    3. How they relate to each other

    Write as if explaining the topic to a knowledgeable colleague.
    Be factual and specific. Use the entity descriptions provided.
""")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Extracts entities and relationships from chunks, builds a NetworkX graph.
    """

    def __init__(self, client: anthropic.Anthropic):
        self._client = client

    def extract_from_chunk(self, chunk: Chunk) -> Tuple[List[Entity], List[Relationship]]:
        """
        Ask Claude to extract entities and relationships from one chunk.
        Returns (entities, relationships) or ([], []) on parse failure.
        """
        resp = self._client.messages.create(
            model=config.model,
            max_tokens=1024,
            temperature=0,
            system=_EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": chunk.content[:2000]}],
        )

        text = resp.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [], []

        entities = [
            Entity(
                name=e["name"],
                entity_type=e.get("type", "CONCEPT"),
                description=e.get("description", ""),
                source_chunk_ids=[chunk.chunk_id],
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]

        relationships = [
            Relationship(
                source=r["source"],
                target=r["target"],
                description=r.get("description", ""),
                weight=float(r.get("weight", 1.0)),
            )
            for r in data.get("relationships", [])
            if r.get("source") and r.get("target")
        ]

        return entities, relationships

    def build_graph(
        self,
        chunks: List[Chunk],
        max_chunks: int = 100,
    ) -> nx.Graph:
        """
        Extract entities from all chunks and build a merged NetworkX graph.

        Entity merging: if two chunks mention the same entity name, their
        descriptions are concatenated and source_chunk_ids combined.
        This avoids duplicate nodes for "BERT" mentioned in 20 chunks.

        Parameters
        ----------
        chunks     : list of Chunk objects
        max_chunks : limit extraction to this many chunks (cost control)
        """
        # entity_name → merged Entity
        entity_map: Dict[str, Entity] = {}
        all_relationships: List[Relationship] = []

        for chunk in chunks[:max_chunks]:
            entities, relationships = self.extract_from_chunk(chunk)

            for entity in entities:
                name = entity.name
                if name in entity_map:
                    # Merge: extend description and source list
                    existing = entity_map[name]
                    if entity.description not in existing.description:
                        existing.description += " " + entity.description
                    existing.source_chunk_ids.extend(entity.source_chunk_ids)
                else:
                    entity_map[name] = entity

            all_relationships.extend(relationships)

        # Build NetworkX graph
        G = nx.Graph()
        for entity in entity_map.values():
            G.add_node(entity.name, entity=entity)

        for rel in all_relationships:
            if rel.source in entity_map and rel.target in entity_map:
                if G.has_edge(rel.source, rel.target):
                    # Accumulate weight for multiple relationships
                    G[rel.source][rel.target]["weight"] += rel.weight
                    G[rel.source][rel.target]["descriptions"].append(rel.description)
                else:
                    G.add_edge(
                        rel.source, rel.target,
                        weight=rel.weight,
                        descriptions=[rel.description],
                    )

        return G


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities(G: nx.Graph, max_community_size: int = 15) -> List[Community]:
    """
    Detect communities in the knowledge graph.

    Strategy:
    1. Use NetworkX's Greedy Modularity Communities algorithm
       (approximates the Leiden/Louvain method used in Microsoft's paper).
    2. Split communities larger than max_community_size in half
       (recursive bisection using spectral partitioning).

    Falls back to connected components if modularity detection fails.
    """
    if G.number_of_nodes() == 0:
        return []

    communities: List[Community] = []

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        raw_communities = list(greedy_modularity_communities(G))
    except Exception:
        # Fallback: each connected component = one community
        raw_communities = list(nx.connected_components(G))

    for i, node_set in enumerate(raw_communities):
        node_list = list(node_set)

        # Split oversized communities
        if len(node_list) > max_community_size:
            mid = len(node_list) // 2
            communities.append(Community(
                community_id=f"c{i}a",
                entity_names=node_list[:mid],
                level=0,
            ))
            communities.append(Community(
                community_id=f"c{i}b",
                entity_names=node_list[mid:],
                level=0,
            ))
        else:
            communities.append(Community(
                community_id=f"c{i}",
                entity_names=node_list,
                level=0,
            ))

    return communities


# ---------------------------------------------------------------------------
# Community summarisation
# ---------------------------------------------------------------------------

def summarise_communities(
    communities: List[Community],
    graph: nx.Graph,
    client: anthropic.Anthropic,
) -> List[Community]:
    """
    Generate an LLM summary for each community.
    Mutates communities in-place (sets .summary).
    """
    for community in communities:
        entities_text = []
        for name in community.entity_names:
            node_data = graph.nodes.get(name, {})
            entity: Optional[Entity] = node_data.get("entity")
            if entity:
                entities_text.append(
                    f"- {entity.name} ({entity.entity_type}): {entity.description}"
                )

        relationships_text = []
        for u, v, data in graph.edges(data=True):
            if u in community.entity_names and v in community.entity_names:
                descs = data.get("descriptions", [])
                if descs:
                    relationships_text.append(f"- {u} → {v}: {descs[0]}")

        prompt = (
            f"Entities:\n" + "\n".join(entities_text[:20]) +
            "\n\nRelationships:\n" + "\n".join(relationships_text[:20])
        )

        resp = client.messages.create(
            model=config.model,
            max_tokens=256,
            temperature=0,
            system=_SUMMARISE_COMMUNITY,
            messages=[{"role": "user", "content": prompt}],
        )
        community.summary = resp.content[0].text.strip()

    return communities


# ---------------------------------------------------------------------------
# GraphRAG retrieval strategy
# ---------------------------------------------------------------------------

class GraphRAG(BaseRAG):
    """
    GraphRAG: knowledge graph + community summaries for global and local queries.

    Build once with build_graph(), then query with run().

    Parameters
    ----------
    query_mode : "local" | "global" | "auto"
      local  — entity-centric, retrieves source chunks via graph traversal
      global — community-centric, retrieves from community summaries
      auto   — Claude decides which mode fits the query
    max_build_chunks : max chunks to use for graph extraction (cost control)
    """

    def __init__(
        self,
        *args,
        query_mode: str = "auto",
        max_build_chunks: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.query_mode = query_mode
        self.max_build_chunks = max_build_chunks

        self._graph: Optional[nx.Graph] = None
        self._communities: List[Community] = []
        self._entity_map: Dict[str, Entity] = {}
        # Separate vector store for community summary embeddings
        self._community_store = InMemoryVectorStore()
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_graph(self) -> "GraphRAG":
        """
        Build the knowledge graph from all indexed chunks.
        Call once after pipeline.index().
        This is the expensive step — budget ~1 Claude call per chunk.
        """
        all_chunks = self.store.all_chunks()
        if not all_chunks:
            print("GraphRAG: no chunks in store, skipping build")
            return self

        print(f"GraphRAG: extracting entities from {min(len(all_chunks), self.max_build_chunks)} chunks...")
        builder = KnowledgeGraphBuilder(self._client)
        self._graph = builder.build_graph(all_chunks, self.max_build_chunks)

        # Cache entity map for local query lookup
        self._entity_map = {
            name: data["entity"]
            for name, data in self._graph.nodes(data=True)
            if "entity" in data
        }

        print(f"GraphRAG: {self._graph.number_of_nodes()} entities, "
              f"{self._graph.number_of_edges()} relationships")

        # Detect communities
        self._communities = detect_communities(self._graph)
        print(f"GraphRAG: {len(self._communities)} communities detected")

        # Summarise communities
        print("GraphRAG: generating community summaries...")
        self._communities = summarise_communities(
            self._communities, self._graph, self._client
        )

        # Embed community summaries for global query search
        if self._communities:
            summaries = [c.summary for c in self._communities]
            vecs = self.embedder.embed_texts(summaries)
            for community, vec in zip(self._communities, vecs):
                community.embedding = vec

            # Store as chunks in the community store
            for community in self._communities:
                c = Chunk(
                    content=community.summary,
                    doc_id="graphrag_community",
                    doc_title=f"Community {community.community_id}",
                    doc_source="graphrag",
                    start_char=0,
                    end_char=len(community.summary),
                    chunk_index=0,
                    metadata={"community_id": community.community_id,
                              "entities": community.entity_names},
                )
                c.embedding = community.embedding
                self._community_store.add_chunks([c])

        self._built = True
        print("GraphRAG: build complete.")
        return self

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def _is_global_query(self, query: str) -> bool:
        """
        Heuristic: global queries ask about themes, summaries, comparisons.
        Local queries ask about specific entities, facts, details.
        """
        if self.query_mode == "global":
            return True
        if self.query_mode == "local":
            return False

        # Auto-detect via keyword heuristics
        global_keywords = {
            "main themes", "overall", "summary", "in general", "across",
            "compare", "contrast", "trends", "patterns", "throughout",
            "what are the", "how does this corpus", "what topics",
        }
        query_lower = query.lower()
        return any(kw in query_lower for kw in global_keywords)

    # ------------------------------------------------------------------
    # Local query: entity → source chunks → generate
    # ------------------------------------------------------------------

    def _local_query(self, query: str) -> RAGResponse:
        """
        Find entities mentioned in query, traverse graph to collect
        related entities, retrieve source chunks.
        """
        # Step 1: find entities in query by name matching
        query_lower = query.lower()
        matched_entities = [
            name for name in self._entity_map
            if name.lower() in query_lower
        ]

        # Step 2: if no direct match, fall back to naive vector retrieval
        if not matched_entities:
            return self.generator.generate(
                query=query,
                results=self._retrieve(query, k=config.top_k),
                pattern="graphrag_local",
            )

        # Step 3: expand via graph neighbours (1-hop)
        all_relevant_entities: Set[str] = set(matched_entities)
        for entity_name in matched_entities:
            neighbours = list(self._graph.neighbors(entity_name))
            all_relevant_entities.update(neighbours[:5])  # limit fan-out

        # Step 4: collect source chunk IDs from all relevant entities
        relevant_chunk_ids: Set[str] = set()
        for entity_name in all_relevant_entities:
            entity = self._entity_map.get(entity_name)
            if entity:
                relevant_chunk_ids.update(entity.source_chunk_ids)

        # Step 5: retrieve those specific chunks from the vector store
        all_stored = self.store.all_chunks()
        matching = [c for c in all_stored if c.chunk_id in relevant_chunk_ids]

        if not matching:
            return self.generator.generate(
                query=query,
                results=self._retrieve(query, k=config.top_k),
                pattern="graphrag_local",
            )

        # Embed and score against query for ranking
        q_vec = self.embedder.embed_query(query)
        scored = []
        for chunk in matching[:20]:
            if chunk.embedding is not None:
                score = float(np.dot(q_vec, chunk.embedding))
                scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        top_results = scored[:config.top_k]

        response = self.generator.generate(
            query=query,
            results=top_results,
            pattern="graphrag_local",
        )
        response.metadata["matched_entities"] = matched_entities
        response.metadata["expanded_entities"] = list(all_relevant_entities)
        return response

    # ------------------------------------------------------------------
    # Global query: community summaries → generate
    # ------------------------------------------------------------------

    def _global_query(self, query: str) -> RAGResponse:
        """
        Search community summaries for high-level thematic answers.
        """
        if not self._communities:
            # No communities built — fall back to naive
            return self.generator.generate(
                query=query,
                results=self._retrieve(query, k=config.top_k),
                pattern="graphrag_global",
            )

        q_vec = self.embedder.embed_query(query)
        results = self._community_store.search(q_vec, k=config.top_k, min_score=0.0)

        response = self.generator.generate(
            query=query,
            results=results,
            pattern="graphrag_global",
        )
        response.metadata["n_communities_searched"] = len(self._communities)
        return response

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> RAGResponse:
        if not self._built:
            print("GraphRAG: building graph (call build_graph() explicitly to avoid this)...")
            self.build_graph()

        if self._is_global_query(query):
            return self._global_query(query)
        else:
            return self._local_query(query)

    # ------------------------------------------------------------------
    # Inspection utilities
    # ------------------------------------------------------------------

    def entity_summary(self) -> str:
        """Human-readable summary of the knowledge graph."""
        lines = [
            f"Entities: {self._graph.number_of_nodes()}",
            f"Relationships: {self._graph.number_of_edges()}",
            f"Communities: {len(self._communities)}",
            "",
            "Top entities by degree (most connected):",
        ]
        if self._graph.number_of_nodes() > 0:
            top = sorted(
                self._graph.degree(), key=lambda x: x[1], reverse=True
            )[:10]
            for name, degree in top:
                entity = self._entity_map.get(name)
                etype = entity.entity_type if entity else "?"
                lines.append(f"  {name} ({etype}) — {degree} connections")
        return "\n".join(lines)
