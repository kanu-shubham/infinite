"""
demo_graph_rag.py — Shows GraphRAG step by step without needing Claude API.

We mock the LLM entity extraction so you can run this locally
and see exactly what happens at each stage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from documents import Document, Chunk, chunk_document
from embeddings import TFIDFEmbedder
from vectorstore import VectorStore
from graph_rag import KnowledgeGraph, Entity, Relationship, Community, detect_communities


# ── Mock data — what the LLM would extract ────────────────────────────────────

MOCK_EXTRACTIONS = {
    0: {
        "entities": [
            Entity("Alice", "Person", "HR Manager", []),
            Entity("Bob", "Person", "VP of HR", []),
        ],
        "relationships": [
            Relationship("Alice", "Bob", "reports to", ""),
        ],
    },
    1: {
        "entities": [
            Entity("Bob", "Person", "VP of HR", []),
            Entity("Carol", "Person", "Finance Manager", []),
        ],
        "relationships": [
            Relationship("Bob", "Carol", "works with", ""),
            Relationship("Bob", "travel requests", "approves", ""),
        ],
    },
    2: {
        "entities": [
            Entity("Carol", "Person", "Finance Manager", []),
            Entity("Bob", "Person", "VP of HR", []),
            Entity("company budget", "Concept", "Annual company budget", []),
        ],
        "relationships": [
            Relationship("Carol", "Bob", "reports to", ""),
            Relationship("Carol", "company budget", "manages", ""),
        ],
    },
    3: {
        "entities": [
            Entity("Dave", "Person", "Lead Engineer", []),
            Entity("PagerDuty", "Tool", "Alerting system for oncall", []),
            Entity("oncall rotation", "Process", "Weekly engineering oncall", []),
        ],
        "relationships": [
            Relationship("Dave", "oncall rotation", "leads", ""),
            Relationship("oncall rotation", "PagerDuty", "uses", ""),
        ],
    },
    4: {
        "entities": [
            Entity("Eve", "Person", "Senior Engineer", []),
            Entity("Dave", "Person", "Lead Engineer", []),
            Entity("CI/CD pipeline", "Tool", "Automated deployment system", []),
        ],
        "relationships": [
            Relationship("Eve", "Dave", "reports to", ""),
            Relationship("Eve", "CI/CD pipeline", "maintains", ""),
        ],
    },
}


def main():
    print("=" * 60)
    print("GraphRAG — Step by Step Demo")
    print("=" * 60)

    # ── Documents ─────────────────────────────────────────────────────────
    docs = [
        Document(
            title="HR Structure",
            source="confluence/hr",
            content="Alice is the HR Manager. She reports to Bob the VP of HR.",
        ),
        Document(
            title="Approval Authority",
            source="confluence/hr/approvals",
            content="Bob approves all travel requests above $500. He works closely with Carol in Finance.",
        ),
        Document(
            title="Finance Team",
            source="confluence/finance",
            content="Carol manages the company budget. She reports to Bob the VP of HR.",
        ),
        Document(
            title="Engineering Oncall",
            source="confluence/eng/oncall",
            content="Dave is the Lead Engineer running the oncall rotation. The team uses PagerDuty for alerts.",
        ),
        Document(
            title="Engineering Team",
            source="confluence/eng",
            content="Eve is a Senior Engineer. She reports to Dave and maintains the CI/CD pipeline.",
        ),
    ]

    # ── Chunk all documents ────────────────────────────────────────────────
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, chunk_size=200, overlap=20)
        all_chunks.extend(chunks)

    print(f"\n[STEP 1] Split {len(docs)} documents into {len(all_chunks)} chunks")

    # ── Fit embedder ───────────────────────────────────────────────────────
    embedder = TFIDFEmbedder(n_components=32)
    embedder.fit([c.content for c in all_chunks])

    # ── STEP 2: Extract entities from each chunk ───────────────────────────
    print("\n[STEP 2] Extract entities and relationships from each chunk")
    print("-" * 50)

    graph = KnowledgeGraph()

    for i, chunk in enumerate(all_chunks):
        extraction = MOCK_EXTRACTIONS.get(i, {"entities": [], "relationships": []})

        for entity in extraction["entities"]:
            entity.source_chunk_ids = [chunk.chunk_id]
            graph.add_entity(entity)

        for rel in extraction["relationships"]:
            rel.chunk_id = chunk.chunk_id
            graph.add_relationship(rel)

        print(f"  Chunk {i}: '{chunk.content[:50]}...'")
        print(f"    Entities: {[e.name for e in extraction['entities']]}")
        print(f"    Relations: {[(r.source, r.relation, r.target) for r in extraction['relationships']]}")

    # ── STEP 3: Show the knowledge graph ──────────────────────────────────
    print(f"\n[STEP 3] Knowledge Graph built")
    print("-" * 50)
    print(f"  Total entities: {len(graph.entities)}")
    print(f"  Total relationships: {len(graph.relationships)}")
    print("\n  Entities:")
    for name, entity in graph.entities.items():
        print(f"    {entity.name} ({entity.type}): {entity.description}")
    print("\n  Relationships:")
    for rel in graph.relationships:
        print(f"    {rel.source} ──{rel.relation}──► {rel.target}")
    print("\n  Adjacency (who connects to whom):")
    for entity, neighbours in graph.adjacency.items():
        print(f"    {entity}: {neighbours}")

    # ── STEP 4: Community detection ────────────────────────────────────────
    print(f"\n[STEP 4] Community Detection (BFS)")
    print("-" * 50)
    communities = detect_communities(graph)
    print(f"  Found {len(communities)} communities:")
    for community in communities:
        print(f"  Community {community.community_id}: {community.entity_names}")

    # ── STEP 5: Summarise communities ─────────────────────────────────────
    print(f"\n[STEP 5] Community Summaries (normally LLM — using mock here)")
    print("-" * 50)

    # Mock summaries — what Claude would generate
    mock_summaries = {
        0: (
            "The HR and Finance leadership community is centred around Bob (VP of HR), "
            "who holds central approval authority. Alice (HR Manager) and Carol (Finance Manager) "
            "both report to Bob. Bob coordinates travel approvals and budget decisions, "
            "working directly with Carol on financial matters."
        ),
        1: (
            "The Engineering team community consists of Dave (Lead Engineer) and Eve (Senior Engineer). "
            "Eve reports to Dave. The team runs a weekly oncall rotation using PagerDuty for alerts "
            "and Eve maintains the CI/CD pipeline for automated deployments."
        ),
    }

    # Embed summaries and store in a separate vector store
    summary_store = VectorStore()

    for community in communities:
        summary = mock_summaries.get(community.community_id, f"Community {community.community_id}")
        community.summary = summary
        print(f"\n  Community {community.community_id} ({community.entity_names}):")
        print(f"  Summary: {summary[:120]}...")

        # Create a synthetic chunk for this community summary
        from documents import Chunk as ChunkClass
        synthetic_chunk = ChunkClass(
            content=summary,
            doc_id=f"community_{community.community_id}",
            doc_title=f"Community {community.community_id} Summary",
            doc_source="graphrag/communities",
            start_char=0,
            end_char=len(summary),
            chunk_index=community.community_id,
        )
        embedding = embedder.embed_query(summary)
        synthetic_chunk.embedding = embedding
        summary_store.add_chunks([synthetic_chunk])

    # ── STEP 6: Query the graph ────────────────────────────────────────────
    print(f"\n[STEP 6] Querying")
    print("-" * 50)

    queries = [
        "Who has authority over travel and budget decisions?",
        "Who manages the engineering oncall process?",
        "What is the relationship between HR and Finance?",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")

        # Regular RAG — searches raw chunks
        q_emb = embedder.embed_query(query)
        raw_store = VectorStore()
        embeddings = embedder.embed_texts([c.content for c in all_chunks])
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb
        raw_store.add_chunks(all_chunks)
        regular_results = raw_store.search(q_emb, top_k=1)

        # GraphRAG — searches community summaries
        graph_results = summary_store.search(q_emb, top_k=1)

        print(f"  Regular RAG top result:")
        if regular_results:
            print(f"    '{regular_results[0].chunk.content[:80]}...'")

        print(f"  GraphRAG top result:")
        if graph_results:
            print(f"    '{graph_results[0].chunk.content[:120]}...'")


if __name__ == "__main__":
    main()
