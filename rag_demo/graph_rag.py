"""
graph_rag.py — GraphRAG implementation.

Flow:
  1. Extract entities and relationships from each chunk (LLM call)
  2. Build a knowledge graph (nodes = entities, edges = relationships)
  3. Detect communities (groups of connected entities)
  4. Summarise each community (LLM call)
  5. At query time, search community summaries instead of raw chunks
"""
import os
import json
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from documents import Chunk
from vectorstore import VectorStore, RetrievalResult


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class Entity:
    """
    A node in the knowledge graph.
    Example: name="Bob", type="Person", description="VP of HR"
    """
    name: str
    type: str                          # Person, Organization, Concept, etc.
    description: str
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """
    An edge in the knowledge graph.
    Example: source="Alice", target="Bob", relation="reports to"
    """
    source: str       # entity name
    target: str       # entity name
    relation: str     # verb describing the relationship
    chunk_id: str     # which chunk this came from


@dataclass
class Community:
    """
    A cluster of related entities.
    Example: [Alice, Bob, Carol] all connected through Bob
    """
    community_id: int
    entity_names: list[str]
    summary: str = ""
    chunk_id: str = ""   # the synthetic chunk we create for this community


# ─────────────────────────────────────────────
# Knowledge Graph
# ─────────────────────────────────────────────

class KnowledgeGraph:
    """
    Stores entities (nodes) and relationships (edges).

    Internal structure:
        entities    : dict[name → Entity]
        relationships: list of Relationship objects
        adjacency   : dict[entity_name → set of connected entity names]
                      Used for community detection.
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        # adjacency list — who is connected to whom
        self.adjacency: dict[str, set[str]] = defaultdict(set)

    def add_entity(self, entity: Entity) -> None:
        name = entity.name.lower()   # normalise to lowercase to merge duplicates

        if name in self.entities:
            # Entity already exists from another chunk — merge descriptions
            existing = self.entities[name]
            if entity.description not in existing.description:
                existing.description += f" | {entity.description}"
            existing.source_chunk_ids.extend(entity.source_chunk_ids)
        else:
            self.entities[name] = entity

    def add_relationship(self, rel: Relationship) -> None:
        self.relationships.append(rel)
        # Build adjacency list for community detection
        src = rel.source.lower()
        tgt = rel.target.lower()
        self.adjacency[src].add(tgt)
        self.adjacency[tgt].add(src)   # undirected for community detection

    def get_entity_names(self) -> list[str]:
        return list(self.entities.keys())


# ─────────────────────────────────────────────
# Community Detection — simple BFS approach
# ─────────────────────────────────────────────

def detect_communities(graph: KnowledgeGraph) -> list[Community]:
    """
    Find groups of connected entities using BFS (Breadth-First Search).

    Think of it like finding islands:
      - Start at any unvisited entity
      - Visit all entities reachable from it (connected by relationships)
      - That group = one community
      - Pick the next unvisited entity → next community
      - Repeat until all entities are visited

    In production you would use the Leiden algorithm (used by Microsoft GraphRAG)
    which finds communities that maximise internal connections (modularity).
    BFS is simpler and good enough for understanding.
    """
    all_entities = set(graph.get_entity_names())
    visited = set()
    communities = []
    community_id = 0

    for entity in all_entities:
        if entity in visited:
            continue   # already part of a community

        # BFS from this entity
        community_members = []
        queue = [entity]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            community_members.append(current)

            # Add all neighbours to the queue
            for neighbour in graph.adjacency.get(current, []):
                if neighbour not in visited:
                    queue.append(neighbour)

        communities.append(Community(
            community_id=community_id,
            entity_names=community_members,
        ))
        community_id += 1

    return communities


# ─────────────────────────────────────────────
# Main GraphRAG class
# ─────────────────────────────────────────────

class GraphRAG:
    """
    Full GraphRAG pipeline.

    Two modes of retrieval:
      local  → search raw chunks (good for specific factual queries)
      global → search community summaries (good for relationship/overview queries)
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        self.model = model
        self.graph = KnowledgeGraph()
        self.communities: list[Community] = []
        # Separate vector store for community summaries
        # (so global queries search summaries, not raw chunks)
        self.summary_store = VectorStore()

    # ─────────────────────────────────────────
    # Step 1 — Extract entities from a chunk
    # ─────────────────────────────────────────

    def extract_entities(self, chunk: Chunk) -> tuple[list[Entity], list[Relationship]]:
        """
        Ask Claude to extract entities and relationships from one chunk.
        Returns structured data we can add to the knowledge graph.
        """
        prompt = f"""Extract entities and relationships from this text.

Text: {chunk.content}

Respond with valid JSON only, no other text:
{{
  "entities": [
    {{"name": "Alice", "type": "Person", "description": "HR Manager"}}
  ],
  "relationships": [
    {{"source": "Alice", "target": "Bob", "relation": "reports to"}}
  ]
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(response.content[0].text)

            entities = [
                Entity(
                    name=e["name"],
                    type=e["type"],
                    description=e["description"],
                    source_chunk_ids=[chunk.chunk_id],
                )
                for e in data.get("entities", [])
            ]

            relationships = [
                Relationship(
                    source=r["source"],
                    target=r["target"],
                    relation=r["relation"],
                    chunk_id=chunk.chunk_id,
                )
                for r in data.get("relationships", [])
            ]

            return entities, relationships

        except Exception:
            return [], []

    # ─────────────────────────────────────────
    # Step 2 — Build graph from all chunks
    # ─────────────────────────────────────────

    def build_graph(self, chunks: list[Chunk]) -> None:
        """
        Process every chunk: extract entities → add to graph.
        This is the expensive step — one LLM call per chunk.
        In production: run in parallel, cache results.
        """
        print(f"  Building graph from {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            entities, relationships = self.extract_entities(chunk)

            for entity in entities:
                self.graph.add_entity(entity)

            for rel in relationships:
                self.graph.add_relationship(rel)

            print(f"  Chunk {i+1}/{len(chunks)}: found {len(entities)} entities, {len(relationships)} relationships")

    # ─────────────────────────────────────────
    # Step 3 — Detect communities
    # ─────────────────────────────────────────

    def detect_and_summarise_communities(self, embedder) -> None:
        """
        1. Find communities using BFS
        2. For each community, generate a natural language summary
        3. Embed that summary and store in summary_store
        """
        self.communities = detect_communities(self.graph)
        print(f"  Detected {len(self.communities)} communities")

        for community in self.communities:
            # Gather all entity descriptions for this community
            entity_details = []
            for name in community.entity_names:
                entity = self.graph.entities.get(name)
                if entity:
                    entity_details.append(f"{entity.name} ({entity.type}): {entity.description}")

            # Gather all relationships within this community
            member_set = set(community.entity_names)
            relevant_rels = [
                f"{r.source} → {r.relation} → {r.target}"
                for r in self.graph.relationships
                if r.source.lower() in member_set and r.target.lower() in member_set
            ]

            # Ask Claude to summarise this community
            summary = self._summarise_community(entity_details, relevant_rels, community.community_id)
            community.summary = summary

            # Create a synthetic Chunk for this community summary
            # so it can be stored in the vector store
            from documents import Chunk as ChunkClass
            import hashlib
            synthetic_chunk = ChunkClass(
                content=summary,
                doc_id=f"community_{community.community_id}",
                doc_title=f"Community {community.community_id} Summary",
                doc_source="graphrag/communities",
                start_char=0,
                end_char=len(summary),
                chunk_index=community.community_id,
            )
            community.chunk_id = synthetic_chunk.chunk_id

            # Embed and store the summary
            embedding = embedder.embed_query(summary)
            synthetic_chunk.embedding = embedding
            self.summary_store.add_chunks([synthetic_chunk])

            print(f"  Community {community.community_id}: {community.entity_names[:3]}...")

    def _summarise_community(
        self,
        entity_details: list[str],
        relationships: list[str],
        community_id: int,
    ) -> str:
        """Ask Claude for a paragraph summarising this community."""
        prompt = f"""Summarise this group of related entities and their relationships
in 2-3 sentences. Focus on what this group does and how they relate to each other.

Entities:
{chr(10).join(entity_details)}

Relationships:
{chr(10).join(relationships)}

Write a concise summary paragraph:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            return f"Community {community_id}: {', '.join(entity_details[:3])}"

    # ─────────────────────────────────────────
    # Step 4 — Query
    # ─────────────────────────────────────────

    def query_global(self, query: str, embedder, top_k: int = 3) -> list[RetrievalResult]:
        """
        Global query — search community summaries.
        Best for: "What is the relationship between X and Y?"
                  "Give me an overview of..."
                  "Who is responsible for...?"
        """
        q_emb = embedder.embed_query(query)
        return self.summary_store.search(q_emb, top_k=top_k)

    def get_graph_stats(self) -> dict:
        return {
            "entities": len(self.graph.entities),
            "relationships": len(self.graph.relationships),
            "communities": len(self.communities),
        }
