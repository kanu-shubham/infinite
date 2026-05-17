"""
demo.py — End-to-end demonstration of the RAG pipeline with ACL.

Shows:
    - How documents are indexed with different permission levels
    - How alice (HR + Engineering) can see both types of documents
    - How bob (Engineering only) cannot see HR documents
    - How the bookkeeper tracks everything
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from documents import Document
from embeddings import TFIDFEmbedder
from pipeline import RAGPipeline
from acl import UserContext


def main():
    print("=" * 60)
    print("RAG DEMO — ACL-aware retrieval")
    print("=" * 60)

    # ── Build a small corpus ───────────────────────────────────────────
    docs = [
        Document(
            title="HR Leave Policy",
            source="confluence/hr/leave-policy",
            content=(
                "Employees are entitled to 20 days of paid annual leave per year. "
                "Leave requests must be submitted at least 2 weeks in advance through "
                "the HR portal. Unused leave up to 10 days can be carried over to the "
                "next year. Sick leave is separate and requires a medical certificate "
                "for absences longer than 3 consecutive days."
            ),
        ),
        Document(
            title="Engineering Deployment Guide",
            source="confluence/eng/deployments",
            content=(
                "All production deployments must go through the CI/CD pipeline. "
                "Direct pushes to main are disabled. Every pull request requires "
                "two approvals and all tests must pass. Deployments happen during "
                "the maintenance window: Tuesday and Thursday 22:00-23:00 UTC. "
                "Rollback is automated via Argo CD health checks."
            ),
        ),
        Document(
            title="Company Travel Policy",
            source="confluence/hr/travel",
            content=(
                "Business travel requires manager approval for expenses above $500. "
                "Economy class is standard for flights under 6 hours. "
                "Business class requires VP approval. Hotel bookings should be made "
                "through the approved travel portal to receive corporate rates."
            ),
        ),
        Document(
            title="Engineering Oncall Runbook",
            source="confluence/eng/oncall",
            content=(
                "Oncall rotation is weekly. The oncall engineer must respond to P1 "
                "alerts within 5 minutes and P2 alerts within 30 minutes. "
                "Use PagerDuty for alerting. All incidents must have a post-mortem "
                "within 48 hours. Escalation path: oncall → team lead → VP Engineering."
            ),
        ),
        Document(
            title="Security Password Policy",
            source="confluence/security/passwords",
            content=(
                "All employees must use passwords with at least 12 characters, "
                "including uppercase, lowercase, numbers and special characters. "
                "Passwords must be rotated every 90 days. MFA is mandatory for "
                "all production system access. Password reuse within the last 10 "
                "passwords is prohibited. Use the company-approved password manager."
            ),
        ),
    ]

    # ── Set up pipeline ────────────────────────────────────────────────
    embedder = TFIDFEmbedder(n_components=32)
    # Fit on all text so the embedder knows the vocabulary
    embedder.fit([d.content for d in docs])

    pipeline = RAGPipeline(embedder=embedder, use_hybrid=True, use_reranker=False)

    # ── Index with permissions ─────────────────────────────────────────
    print("\n[INDEXING]")
    for doc in docs:
        perms = set()
        if "hr" in doc.source:
            perms = {"hr_docs"}
        elif "eng" in doc.source:
            perms = {"engineering_docs"}
        elif "security" in doc.source:
            perms = {"security_docs"}

        chunks = pipeline.index_document(doc, permissions=perms)
        print(f"  {doc.title}: {len(chunks)} chunk(s), permissions={perms or 'public'}")

    # ── Define users ───────────────────────────────────────────────────
    alice = UserContext(user_id="alice", permissions={"hr_docs", "engineering_docs"})
    bob   = UserContext(user_id="bob",   permissions={"engineering_docs"})

    # ── Demo query ─────────────────────────────────────────────────────
    query = "How many days of leave do employees get?"
    print(f"\n[QUERY] '{query}'")
    print("-" * 40)

    for user in [alice, bob]:
        # We call the vector store directly (skip LLM to avoid needing API key)
        from hybrid import HybridRetriever
        from acl import can_access

        retriever = HybridRetriever(pipeline.bm25_index, pipeline.vector_store, pipeline.embedder)
        results = retriever.search(query, top_k=10)
        filtered = [r for r in results if can_access(user, r.chunk.metadata.get("permissions", set()))]

        print(f"\n  User: {user.user_id} | permissions: {user.permissions}")
        if filtered:
            top = filtered[0]
            print(f"  Top result: '{top.chunk.doc_title}' (score={top.score:.4f})")
            print(f"  Preview: {top.chunk.content[:120]}...")
        else:
            print("  No accessible results — ACL blocked all chunks.")

    # ── Bookkeeper audit log ───────────────────────────────────────────
    print("\n[BOOKKEEPER]")
    pipeline.bookkeeper.log_query("alice", query, 3, "hybrid")
    pipeline.bookkeeper.log_query("bob", query, 0, "hybrid")

    cursor = pipeline.bookkeeper.conn.cursor()
    cursor.execute("SELECT user_id, query, num_results, queried_at FROM query_log")
    for row in cursor.fetchall():
        print(f"  {row[0]} | '{row[1]}' | {row[2]} results | {row[3]}")

    # ── Upsert demo ────────────────────────────────────────────────────
    print("\n[UPSERT — updating HR Leave Policy]")
    updated_hr = Document(
        title="HR Leave Policy",
        source="confluence/hr/leave-policy",
        content=(
            "UPDATED: Employees are now entitled to 25 days of paid annual leave per year. "
            "The HR portal will be updated by end of Q1. All other policies remain unchanged."
        ),
    )
    new_chunks = pipeline.upsert_document(updated_hr, permissions={"hr_docs"})
    print(f"  Upserted {len(new_chunks)} new chunk(s) for doc_id={updated_hr.doc_id}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
