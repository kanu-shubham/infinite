"""
demo.py — Run the complete RAG pipeline end-to-end (fully offline)
===================================================================
Walkthrough:
  1. Corpus loading
  2. Three chunking strategies compared
  3. BM25 vs Dense vs Hybrid retrieval
  4. RAGAS-style evaluation

Run:  python demo.py
"""

import sys, time
sys.path.insert(0, "/home/user/infinite/rag")

from corpus.corpus_loader import load_ai_corpus
from corpus.chunkers import (
    RecursiveCharacterChunker,
    SemanticChunker,
    ParentDocumentChunker,
)
from embeddings.local_embedder import LocalTFIDFEmbedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.pipeline import RAGPipeline, PipelineConfig
from evaluation.ragas_eval import (
    SelfContainedRAGAS, EvalSample,
    build_test_set, print_eval_report,
)


def sep(title="", w=68):
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * w)


# ─── 1. LOAD CORPUS ───────────────────────────────────────────────────────────
sep("1 · CORPUS")
corpus = load_ai_corpus()
print(f"Loaded {len(corpus)} documents:\n")
for doc in corpus:
    print(f"  {doc.doc_id}  '{doc.title}'  ({len(doc.text.split())} words)")


# ─── 2. CHUNKING STRATEGIES ───────────────────────────────────────────────────
sep("2 · CHUNKING STRATEGIES")
doc = corpus[1]   # Transformer doc — nicely structured
print(f"\nDocument: '{doc.title}' — {len(doc.text)} chars\n")

# Strategy A: Recursive character splitter
rc = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=40)
rc_chunks = rc.chunk_document(doc)
sizes = [len(c.text) for c in rc_chunks]
print(f"[A] RecursiveCharacter  chunk_size=400, overlap=40")
print(f"    → {len(rc_chunks)} chunks, sizes: {sizes}")
print(f"    First chunk preview: '{rc_chunks[0].text[:110]}...'\n")

# Strategy B: Semantic (needs embedder — we show config; actually runs as recursive fallback
#             when embed_fn=None; with embed_fn it splits on topic shifts)
sem = SemanticChunker(embed_fn=None, breakpoint_threshold=0.3)
sem_chunks = sem.chunk_document(doc)
print(f"[B] SemanticChunker  threshold=0.3  (embed_fn=None → recursive fallback)")
print(f"    → {len(sem_chunks)} chunks")
print(f"    TIP: Pass embed_fn=embedder.embed_documents to split on topic shifts.\n")

# Strategy C: Parent-document
pd = ParentDocumentChunker(parent_chunk_size=600, child_chunk_size=150)
parents, children = pd.chunk_document(doc)
print(f"[C] ParentDocument  parent=600 chars, child=150 chars")
print(f"    → {len(parents)} parents, {len(children)} children")
for p in parents[:2]:
    kids = [c for c in children if c.parent_id == p.chunk_id]
    print(f"    {p.chunk_id} ({len(p.text)} chars) → {len(kids)} children")
    for k in kids[:2]:
        print(f"      {k.chunk_id}: '{k.text[:70]}...'")


# ─── 3. BUILD INDICES ─────────────────────────────────────────────────────────
sep("3 · INDEXING  (BM25 + Dense)")

# Chunk the whole corpus
all_chunks = rc.chunk_corpus(corpus)
print(f"\nTotal chunks across corpus: {len(all_chunks)}")

# Fit local TF-IDF + SVD embedder on all chunk texts (fully offline)
print("\nFitting offline TF-IDF+SVD embedder on corpus chunks...")
all_texts = [c.text for c in all_chunks]
embedder = LocalTFIDFEmbedder(n_components=128)
embedder.fit(all_texts)

# BM25 index
bm25 = BM25Retriever()
bm25.index(all_chunks)

# Dense (Qdrant in-memory)
dense = DenseRetriever(embedder)
dense.index(all_chunks)

print("\nBoth indices ready.")


# ─── 4. RETRIEVAL COMPARISON ──────────────────────────────────────────────────
sep("4 · BM25  vs  Dense  vs  Hybrid")

test_queries = [
    ("keyword",  "BM25 TF-IDF k1 b parameter term frequency saturation"),
    ("semantic", "how neural networks learn from examples using gradient"),
    ("mixed",    "RAGAS evaluation faithfulness context precision metrics"),
]

hybrid = HybridRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    reranker=None,        # no network → no cross-encoder; RRF alone is shown
    bm25_top_k=30,
    dense_top_k=30,
    fusion_top_k=15,
    final_top_k=5,
)

for qtype, query in test_queries:
    print(f"\n── Query type: {qtype}")
    print(f"   '{query}'")

    b_res = bm25.search(query, top_k=3)
    d_res = dense.search(query, top_k=3)
    h_res = hybrid.search(query)

    print(f"\n  BM25 top-3:")
    for r in b_res:
        print(f"    [{r.rank}] {r.score:6.3f}  [{r.metadata.get('title','')[:38]}]")

    print(f"\n  Dense top-3:")
    for r in d_res:
        print(f"    [{r.rank}] {r.score:.4f}  [{r.metadata.get('title','')[:38]}]")

    print(f"\n  Hybrid top-5 (RRF fusion):")
    bm25_ids  = {r.chunk_id for r in b_res}
    dense_ids = {r.chunk_id for r in d_res}
    for r in h_res:
        src = []
        if r.chunk_id in bm25_ids:  src.append("BM25")
        if r.chunk_id in dense_ids: src.append("Dense")
        tag = "+".join(src) if src else "fusion-only"
        print(f"    [{r.rank}] rrf={r.score:.5f}  [{tag:12s}]  "
              f"[{r.metadata.get('title','')[:30]}]")


# ─── 5. FULL PIPELINE QUERY ───────────────────────────────────────────────────
sep("5 · FULL PIPELINE  (retrieve → generate)")

config = PipelineConfig(
    chunk_size=400,
    chunk_overlap=50,
    use_reranker=False,   # cross-encoder needs network; disabled for offline demo
    final_top_k=3,
)
pipeline = RAGPipeline(config)
pipeline.build(corpus, verbose=True)

questions = [
    "What is the self-attention mechanism and how does it differ from RNNs?",
    "Which embedding models rank highest on MTEB?",
    "How do you evaluate a RAG pipeline?",
    "What is the difference between BM25 and dense retrieval?",
]

for q in questions:
    result = pipeline.query(q)
    print(f"\n{'='*68}")
    print(f"Q: {result.question}")
    print(f"{'─'*68}")
    for i, (ctx, src) in enumerate(zip(result.contexts, result.context_sources), 1):
        print(f"[Context {i}] {src}")
        print(f"  '{ctx[:160]}{'...' if len(ctx)>160 else ''}'")
    print(f"{'─'*68}")
    print(f"A: {result.answer}")
    print(f"   ({result.num_chunks_retrieved} chunks retrieved in "
          f"{result.retrieval_time_ms:.0f}ms)")


# ─── 6. RAGAS EVALUATION ──────────────────────────────────────────────────────
sep("6 · RAGAS EVALUATION  (embedding-proxy metrics)")

test_set = build_test_set()
print(f"\nEvaluating on {len(test_set)} test questions...\n")

eval_samples = []
for case in test_set:
    res = pipeline.query(case["question"])
    eval_samples.append(EvalSample(
        question=case["question"],
        contexts=res.contexts,
        answer=res.answer,
        ground_truth=case["ground_truth"],
    ))

evaluator = SelfContainedRAGAS(pipeline.embedder)
eval_results = evaluator.evaluate_dataset(eval_samples)
print_eval_report(eval_results)


# ─── 7. CHUNKING IMPACT EXPERIMENT ───────────────────────────────────────────
sep("7 · EXPERIMENT: Chunk size impact on retrieval")

print("\nBuilding two mini-pipelines with different chunk sizes...")
results_table = []

for chunk_size in [200, 400, 700]:
    cfg = PipelineConfig(chunk_size=chunk_size, chunk_overlap=40,
                         use_reranker=False, final_top_k=3)
    p = RAGPipeline(cfg)
    p.build(corpus, verbose=False)

    # Quick evaluation on first 3 test questions
    samples = []
    for case in test_set[:3]:
        r = p.query(case["question"])
        samples.append(EvalSample(
            question=case["question"],
            contexts=r.contexts,
            answer=r.answer,
            ground_truth=case["ground_truth"],
        ))
    ev = SelfContainedRAGAS(p.embedder)
    er = ev.evaluate_dataset(samples)
    agg = er["aggregate"]
    chunks_n = len(RecursiveCharacterChunker(chunk_size=chunk_size,
                                             chunk_overlap=40).chunk_corpus(corpus))
    results_table.append((chunk_size, chunks_n, agg))
    print(f"  chunk_size={chunk_size:4d}  → {chunks_n:3d} chunks  "
          f"ragas={agg['ragas_score']:.3f}  "
          f"precision={agg['context_precision']:.3f}  "
          f"recall={agg['context_recall']:.3f}")

print("\nINSIGHT:")
print("  Smaller chunks → more chunks → higher precision (less noise per chunk)")
print("  Larger chunks  → fewer chunks → higher recall (more context captured)")
print("  Sweet spot for Q&A is typically 300-500 chars / 100-150 tokens.")


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
sep("DONE")
print("""
What was built:
  corpus/corpus_loader.py    10 synthetic AI/ML documents
  corpus/chunkers.py         3 strategies: Recursive, Semantic, Parent-Document
  embeddings/embedder.py     Neural embedder (sentence-transformers, needs network)
  embeddings/local_embedder.py  Offline TF-IDF+SVD embedder (runs anywhere)
  retrieval/bm25_retriever.py   BM25 keyword search
  retrieval/dense_retriever.py  Dense ANN search via Qdrant
  retrieval/hybrid_retriever.py BM25+Dense RRF fusion + cross-encoder reranker
  retrieval/pipeline.py      End-to-end RAG pipeline
  evaluation/ragas_eval.py   RAGAS metrics (faithfulness/relevancy/precision/recall)
  THEORY.md                  Deep-dive theory guide for all concepts

Next steps to level up:
  1. Plug in a real LLM:  replace SimpleGenerator with OpenAI / Anthropic / Ollama
  2. Better embeddings:   set TRANSFORMERS_OFFLINE=0, use BAAI/bge-large-en-v1.5
  3. Real RAGAS:          pip install ragas langchain-openai  (LLM-as-judge)
  4. Try SemanticChunker: pass embedder.embed_documents as embed_fn
  5. Scale up:            swap QdrantClient(\":memory:\") for a real Qdrant server
""")
