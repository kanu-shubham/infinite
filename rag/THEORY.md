# RAG (Retrieval-Augmented Generation) — Complete Theory Guide

## Table of Contents
1. [What is RAG?](#1-what-is-rag)
2. [Embedding Models & MTEB](#2-embedding-models--mteb)
3. [Chunking Strategies](#3-chunking-strategies)
4. [Vector Databases](#4-vector-databases)
5. [Hybrid Search: BM25 + Dense](#5-hybrid-search-bm25--dense)
6. [Reranking](#6-reranking)
7. [RAGAS Evaluation](#7-ragas-evaluation)
8. [Architecture Overview](#8-architecture-overview)

---

## 1. What is RAG?

RAG = Retrieval-Augmented Generation.

**Problem it solves:** LLMs have a knowledge cutoff and can't know about YOUR documents.

**Core idea:**
```
User Query
    │
    ▼
[Retriever] ──── searches ───► [Document Store]
    │                                │
    │         returns top-K docs     │
    ◄────────────────────────────────┘
    │
    ▼
[LLM] receives: query + retrieved docs → generates answer
```

**Why retrieval beats stuffing all docs in context:**
- Cost: retrieving 5 chunks is cheaper than 10,000 tokens of context
- Quality: LLMs are distracted by irrelevant context
- Scale: you can have millions of documents

---

## 2. Embedding Models & MTEB

### What is an Embedding?

An embedding converts text → a dense vector of numbers (e.g., 768 dimensions).

```
"The cat sat on the mat"  →  [0.12, -0.34, 0.89, ...]  (768 numbers)
"A kitten rested on a rug" →  [0.11, -0.31, 0.91, ...]  (768 numbers)
```

Semantically similar text → vectors that are close together (high cosine similarity).

### The Math: Cosine Similarity

```
similarity(A, B) = (A · B) / (|A| × |B|)

Range: -1 (opposite) to +1 (identical)
```

### MTEB Leaderboard

**MTEB** = Massive Text Embedding Benchmark (https://huggingface.co/spaces/mteb/leaderboard)

It benchmarks models on 8 tasks: Retrieval, Clustering, Classification, etc.

**Top models (as of 2025):**

| Model | Dims | Size | MTEB Score | Notes |
|-------|------|------|------------|-------|
| `text-embedding-3-large` (OpenAI) | 3072 | API | 64.6 | Best overall, paid |
| `BAAI/bge-m3` | 1024 | 570MB | 62.8 | Best open-source multilingual |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | 64.2 | Best open English |
| `thenlper/gte-large` | 1024 | 670MB | 63.1 | Great balance |
| `BAAI/bge-small-en-v1.5` | 384 | 133MB | 62.2 | **Best small model** |
| `all-MiniLM-L6-v2` | 384 | 80MB | 56.3 | Fastest, good baseline |

**How to choose:**
- Dev/learning → `all-MiniLM-L6-v2` (tiny, fast)
- Production English → `BAAI/bge-large-en-v1.5`
- Production multilingual → `BAAI/bge-m3`
- Best quality (paid) → OpenAI `text-embedding-3-large`

**Key insight:** Bigger dims ≠ always better. bge-small (384d) often beats larger models on specific tasks.

---

## 3. Chunking Strategies

### Why Chunk?

Models have token limits. You can't embed 100-page PDFs as one vector — the meaning gets diluted.

### Strategy 1: Fixed-Size / Recursive Character Splitting

```
Original: "The quick brown fox jumped over the lazy dog. It was a sunny day..."

chunk_size=100, overlap=20

Chunk 1: "The quick brown fox jumped over the lazy dog. It was a"
Chunk 2: "a sunny day..."  (20 chars overlap ensures continuity)
```

**Pros:** Simple, predictable  
**Cons:** May split mid-sentence, mid-idea

**LangChain's RecursiveCharacterTextSplitter** tries these separators in order:
`["\n\n", "\n", ". ", " ", ""]` — prefers paragraph breaks, then sentences, then words.

### Strategy 2: Semantic Chunking

Instead of splitting by character count, split where the **meaning changes**.

```
Algorithm:
1. Split text into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentences
4. When similarity DROPS below threshold → new chunk boundary
```

```
Sentence 1: "Dogs are great pets."  → embed → [0.2, 0.8, ...]
Sentence 2: "They are loyal."        → embed → [0.19, 0.81, ...] ← similar, same chunk
Sentence 3: "The stock market fell." → embed → [-0.5, 0.1, ...]  ← VERY different → NEW CHUNK
```

**Pros:** Chunks contain coherent ideas  
**Cons:** Slower (requires embedding during chunking), non-deterministic size

### Strategy 3: Parent-Document Retrieval

A two-level approach:

```
Parent Doc (512 tokens):
"Machine learning is a subset of AI that enables systems 
to learn from data without being explicitly programmed..."

↓ split into child chunks

Child 1 (128 tokens): "Machine learning is a subset of AI..."
Child 2 (128 tokens): "...that enables systems to learn from data..."
Child 3 (128 tokens): "...without being explicitly programmed..."
```

**Retrieval flow:**
1. Embed and store the **small child chunks** in vector DB
2. When a child chunk matches a query → return the **parent document**
3. LLM gets MORE context while retrieval is PRECISE

**Why?** Small chunks = more precise semantic match. Large parent = enough context for LLM.

### Chunk Size Guidelines

| Use Case | Chunk Size | Overlap |
|----------|-----------|---------|
| Q&A over docs | 256–512 tokens | 50 tokens |
| Summarization | 512–1024 tokens | 100 tokens |
| Code retrieval | 1 function/class | None |
| Parent-child | Child: 128, Parent: 512 | 20 tokens |

---

## 4. Vector Databases

### What they store

```
Document: "Paris is the capital of France"
↓ embed ↓
Vector: [0.12, -0.43, 0.89, ... 768 dims]

Vector DB stores: { id, vector, metadata: {text, source, page, ...} }
```

### Similarity Search (ANN — Approximate Nearest Neighbor)

Brute force: compare query to ALL vectors. O(N) — too slow at scale.

**ANN algorithms:**
- **HNSW** (Hierarchical Navigable Small World) — graph-based, used by Qdrant, Weaviate
- **IVF** (Inverted File Index) — cluster-based, used by FAISS
- **LSH** (Locality Sensitive Hashing) — hash-based, simple

HNSW is the gold standard: fast queries (ms), high recall, but more memory.

### Comparison: Qdrant vs Pinecone vs pgvector

| Feature | Qdrant | Pinecone | pgvector |
|---------|--------|----------|----------|
| Type | Dedicated vector DB | Managed cloud | Postgres extension |
| Deployment | Self-hosted / Cloud | Cloud only | Your Postgres |
| Filtering | ✅ Rich payload filters | ✅ Metadata filters | ✅ SQL WHERE |
| Hybrid search | ✅ Native sparse+dense | ✅ Built-in | ⚠️ Manual |
| Scalability | ✅ Distributed | ✅ Fully managed | ⚠️ Limited |
| Cost | Free self-host | Pay per vector | Free (infra cost) |
| Best for | Most production use cases | Simplicity at scale | Existing Postgres users |

**We use Qdrant** in this project — it runs locally, has excellent Python SDK, supports hybrid search natively.

### How HNSW Works (simplified)

```
Layer 2 (sparse, long-range): A ←————→ E
Layer 1 (medium):             A←→B←→C←→D←→E
Layer 0 (dense, all nodes):   A↔B↔C↔D↔E↔F↔G...

Query comes in → start at top layer → navigate down
→ each layer narrows the search → very fast
```

---

## 5. Hybrid Search: BM25 + Dense

### BM25 — The Classic Keyword Algorithm

BM25 (Best Match 25) is TF-IDF on steroids. Used by Elasticsearch, Solr.

**Formula:**
```
Score(D, Q) = Σ IDF(qᵢ) × [f(qᵢ,D) × (k₁+1)] / [f(qᵢ,D) + k₁×(1-b+b×|D|/avgdl)]

Where:
- f(qᵢ,D) = term frequency in doc D
- IDF(qᵢ) = log((N-n+0.5)/(n+0.5)) — rarer terms score higher
- |D| = document length
- avgdl = average document length
- k₁=1.5, b=0.75 (tunable parameters)
```

**Key insight:** BM25 rewards rare terms (IDF) but caps term frequency (TF saturation). "the the the the" doesn't keep boosting score.

**BM25 strengths:**
- Exact keyword match ("GPT-4o" finds "GPT-4o", not "GPT-3")
- No embeddings needed, ultra fast
- Great for rare terms, product names, codes

**BM25 weaknesses:**
- No semantic understanding ("car" ≠ "automobile")
- Word order ignored
- No synonym awareness

### Dense Retrieval Strengths/Weaknesses

**Dense strengths:**
- Semantic similarity ("car" ≈ "automobile")
- Handles paraphrasing
- Language-agnostic (with multilingual models)

**Dense weaknesses:**
- Can miss exact keyword matches
- Embedding model bias
- Requires GPU for large scale

### Hybrid = Best of Both Worlds

```
Query: "What is the capital of France?"

BM25 results:     [doc5(score=8.2), doc12(score=6.1), doc3(score=5.0)]
Dense results:    [doc5(0.92), doc8(0.87), doc3(0.84)]

Fusion → [doc5, doc3, doc8, doc12, ...]
```

### Reciprocal Rank Fusion (RRF)

The standard way to combine ranked lists:

```
RRF_score(d) = Σ 1/(k + rank(d, list_i))
                i

Where k=60 (constant, prevents top ranks from dominating)

Example:
doc5 in BM25: rank=1 → 1/(60+1) = 0.0164
doc5 in Dense: rank=1 → 1/(60+1) = 0.0164
doc5 total: 0.0328

doc8 in BM25: not found → 0
doc8 in Dense: rank=2 → 1/(60+2) = 0.0161
doc8 total: 0.0161
```

**Why RRF?** Score scales are incompatible (BM25=8.2, Dense=0.92). RRF uses only ranks, making fusion scale-free.

---

## 6. Reranking

### Why Rerank?

Initial retrieval (BM25/dense) is optimized for SPEED (ANN, not exact).

Reranking is a **second-stage** using a slower but more powerful **cross-encoder**.

```
Stage 1: Bi-encoder (fast, ~1ms per query)
  Query → [embed] → vector
  Doc   → [embed] → vector
  score = cosine(query_vec, doc_vec)

Stage 2: Cross-encoder (slower, ~50ms per query)
  [Query + Doc] → [full attention model] → relevance score 0-1

Cross-encoder "sees" the query AND document together:
  - Can catch exact phrases
  - Can understand context
  - Much more accurate
```

### Popular Reranking Models

| Model | Notes |
|-------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, good quality |
| `BAAI/bge-reranker-large` | Best open-source |
| `Cohere Rerank` | Best commercial API |
| `Jina Reranker` | Good multilingual |

### The Full Pipeline

```
Query
  │
  ├──► BM25 → top 50 candidates
  │
  ├──► Dense search → top 50 candidates
  │
  └──► RRF Fusion → top 20 candidates
            │
            ▼
       Cross-Encoder Reranker
            │
            ▼
        Top 5 results → LLM → Answer
```

---

## 7. RAGAS Evaluation

RAGAS = **RA**G **A**ssessment **S**core

It evaluates RAG pipelines WITHOUT needing human labels (uses LLM as judge).

### 4 Core Metrics

#### 1. Faithfulness (0–1)
*Does the answer only contain info from the retrieved context?*

```
Context: "Paris is in France. France has 67 million people."
Answer:  "Paris is in France and has a population of 10 million."

Faithfulness = 2/3 ≈ 0.67
(2 of 3 claims are supported, "10 million" is hallucinated)
```

#### 2. Answer Relevancy (0–1)
*Does the answer address the question asked?*

```
Question: "What is the capital of France?"
Answer: "France is a beautiful country with great wine."

Relevancy ≈ 0.1 (didn't answer the question)
```

Method: Generate N alternative questions from the answer → check if they match original question.

#### 3. Context Precision (0–1)
*Are the retrieved documents actually relevant?*

Penalizes retrieving irrelevant chunks. A retriever that fetches 10 chunks, 8 of which are irrelevant, scores low.

#### 4. Context Recall (0–1)
*Were all necessary facts present in retrieved context?*

```
Ground truth answer needs facts: [F1, F2, F3]
Retrieved context contains: [F1, F2]

Recall = 2/3 ≈ 0.67
```

### How RAGAS Uses LLM-as-Judge

```python
# RAGAS uses GPT-4 / Claude to evaluate:
prompt = """
Given: context, question, answer
Task: List all factual claims in the answer.
For each claim, is it supported by the context?
Output JSON: {"supported": [...], "unsupported": [...]}
"""
# Then: faithfulness = len(supported) / (len(supported) + len(unsupported))
```

---

## 8. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                         │
│                                                              │
│  Documents → Chunker → Embedder → Qdrant (dense vectors)    │
│                    └──────────→ BM25 Index (sparse)          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                            │
│                                                              │
│  Query → Embed → Qdrant dense search ──┐                     │
│        → Tokenize → BM25 search ───────┤                     │
│                                        ▼                     │
│                              RRF Fusion (top 20)             │
│                                        │                     │
│                                        ▼                     │
│                            Cross-Encoder Reranker            │
│                                        │                     │
│                                        ▼                     │
│                                Top 5 Chunks → LLM            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION (RAGAS)                         │
│                                                              │
│  Test set → Pipeline → {context, answer, question}           │
│           → RAGAS metrics: faithfulness, relevancy,          │
│             context_precision, context_recall                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

```python
# The mental model:
# 1. CHUNK your docs (not too big, not too small)
# 2. EMBED each chunk (semantic fingerprint)  
# 3. STORE in vector DB with metadata
# 4. At query time: RETRIEVE with hybrid (BM25 + dense)
# 5. RERANK the candidates with cross-encoder
# 6. GENERATE answer with LLM + retrieved context
# 7. EVALUATE with RAGAS
```
