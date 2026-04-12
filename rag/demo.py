#!/usr/bin/env python3
"""
RAG Patterns Demo Script

Demonstrates all 6 RAG patterns on a small curated set of questions,
with explanations of what each pattern does differently.

Run:
  export ANTHROPIC_API_KEY=sk-ant-...
  python demo.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.documents import Document
from src.pipeline import RAGPipeline

SEPARATOR = "─" * 70


def load_sample_docs():
    """Load the built-in sample knowledge base."""
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(docs_dir, fname)
        content = open(path).read().strip()
        lines = content.splitlines()
        title = fname.replace("_", " ").replace(".txt", "").title()
        source = fname
        if lines and lines[0].startswith("Title:"):
            title = lines[0][len("Title:"):].strip()
        if len(lines) > 1 and lines[1].startswith("Source:"):
            source = lines[1][len("Source:"):].strip()
        docs.append(Document(content=content, title=title, source=source))
    return docs


def run_demo():
    try:
        config.validate()
    except ValueError as e:
        print(f"\n  Error: {e}\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  Production RAG System — Pattern Comparison Demo")
    print("=" * 70)

    # ---- Index ----
    docs = load_sample_docs()
    pipeline = RAGPipeline()
    pipeline.index(docs)
    print(f"\nKnowledge base: {len(docs)} documents, {pipeline.chunk_count} chunks\n")

    # ---- Demo 1: Naive RAG vs HyDE ----
    print(SEPARATOR)
    print("DEMO 1: Naive RAG vs HyDE — Vocabulary Mismatch")
    print(SEPARATOR)
    print("""
Naive RAG embeds the raw question; HyDE first generates a hypothetical
answer document, then embeds THAT for retrieval.  HyDE typically finds
more relevant documents when query vocabulary differs from document vocabulary.
""")

    query = "What technique from 2022 generates a fake answer to improve search?"
    print(f"Query: {query!r}\n")

    for pattern in ("naive", "hyde"):
        result = pipeline.query(query, pattern=pattern)
        print(f"[{pattern.upper()}]")
        print(f"  Answer: {result.answer[:300]}...")
        if result.citations:
            print(f"  Top source: {result.citations[0].doc_title}")
        print()
        time.sleep(1)

    # ---- Demo 2: Self-RAG ----
    print(SEPARATOR)
    print("DEMO 2: Self-RAG — Adaptive Retrieval")
    print(SEPARATOR)
    print("""
Self-RAG decides whether retrieval is even needed.  For simple factual
questions it often skips retrieval entirely.  For complex questions it
retrieves and then verifies that the answer is actually grounded.
""")

    simple_q = "What does the acronym GPU stand for?"
    complex_q = "What specific paper introduced RAPTOR and in what year?"

    for q in (simple_q, complex_q):
        print(f"Query: {q!r}")
        result = pipeline.query(q, pattern="self_rag")
        skipped = result.metadata.get("retrieval_skipped", False)
        print(f"  Retrieval skipped: {skipped}")
        print(f"  Answer: {result.answer[:250]}...")
        print()
        time.sleep(1)

    # ---- Demo 3: Multi-hop Retrieval ----
    print(SEPARATOR)
    print("DEMO 3: Multi-hop Retrieval — Chained Reasoning")
    print(SEPARATOR)
    print("""
Multi-hop iteratively refines retrieval queries.  After seeing the first
set of documents, it asks: "what do I still need to find?"  This enables
answering questions that require chaining facts across documents.
""")

    mh_query = (
        "Which optimiser was introduced in 2015 that is commonly used to "
        "train transformer models like BERT?"
    )
    print(f"Query: {mh_query!r}\n")
    result = pipeline.query(mh_query, pattern="multi_hop")
    hops = result.metadata.get("hops", 1)
    hop_log = result.metadata.get("hop_log", [])
    print(f"  Hops taken: {hops}")
    for i, hop in enumerate(hop_log):
        if hop.get("next_query") and hop["next_query"] != "DONE":
            print(f"  Hop {i+1} follow-up: {hop['next_query']!r}")
    print(f"  Answer: {result.answer[:300]}...")
    print()
    time.sleep(1)

    # ---- Demo 4: Query Decomposition ----
    print(SEPARATOR)
    print("DEMO 4: Query Decomposition — Compound Questions")
    print(SEPARATOR)
    print("""
Query decomposition breaks a compound question into focused sub-queries,
retrieves for each independently, then merges with Reciprocal Rank Fusion.
This outperforms naive RAG on multi-aspect questions.
""")

    comp_query = (
        "Compare how BERT and GPT differ in their architecture, "
        "training objective, and typical use cases."
    )
    print(f"Query: {comp_query!r}\n")
    result = pipeline.query(comp_query, pattern="query_decomp")
    sub_queries = result.metadata.get("sub_queries", [])
    print(f"  Decomposed into {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"    {i}. {sq}")
    print(f"\n  Answer: {result.answer[:400]}...")
    print()
    time.sleep(1)

    # ---- Demo 5: No-result fallback ----
    print(SEPARATOR)
    print("DEMO 5: Edge Case — No Relevant Results")
    print(SEPARATOR)
    print("""
When no retrieved document scores above the similarity threshold,
the system falls back gracefully rather than hallucinating from context.
""")

    oor_query = "What are the regulations for quantum computing chip export controls?"
    print(f"Query: {oor_query!r}\n")
    result = pipeline.query(oor_query, pattern="naive")
    print(f"  Fallback triggered: {result.fallback}")
    print(f"  Answer: {result.answer[:300]}...")
    print()

    # ---- Demo 6: RAPTOR (skip in quick demo due to build time) ----
    print(SEPARATOR)
    print("DEMO 6: RAPTOR — Hierarchical Tree Retrieval")
    print(SEPARATOR)
    print("""
RAPTOR builds a tree of document summaries at multiple levels.
Level 0 = raw chunks. Level 1 = cluster summaries. Level 2+ = meta-summaries.
This allows retrieving both fine-grained details AND high-level concepts.

(Skipped in this demo — use python main.py --pattern raptor for full run)
""")

    print(SEPARATOR)
    print("Demo complete!  Next steps:")
    print("  python main.py                          # interactive mode")
    print("  python main.py --compare -q 'your q'   # compare all patterns")
    print("  python main.py --evaluate -q 'your q'  # include metrics")
    print(SEPARATOR)


if __name__ == "__main__":
    run_demo()
