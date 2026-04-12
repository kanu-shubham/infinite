"""
corpus_loader.py
================
Loads and creates a sample document corpus for learning.

THEORY:
  A "corpus" is just a collection of documents.
  In production this would be PDFs, web pages, database records, etc.

  Here we create a rich synthetic corpus about AI/ML topics so we can
  demonstrate retrieval meaningfully without needing external files.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Document:
    """
    A raw document before chunking.

    Attributes:
        doc_id   : unique identifier
        title    : human-readable title
        text     : full document content
        source   : where it came from (file, url, etc.)
        metadata : any extra fields (author, date, etc.)
    """
    doc_id: str
    title: str
    text: str
    source: str = "synthetic"
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Document(id={self.doc_id!r}, title={self.title!r}, chars={len(self.text)})"


def load_ai_corpus() -> List[Document]:
    """
    Returns a synthetic corpus of 10 documents about AI/ML.
    Each document is long enough to require chunking (~300-600 words).

    WHY SYNTHETIC?
    --------------
    Real corpora come from PDFs, web scraping, databases.
    For learning, synthetic docs let us know the ground truth answers,
    making evaluation easier to understand.
    """

    docs = [
        Document(
            doc_id="doc_001",
            title="Introduction to Neural Networks",
            text="""
Neural networks are computational models inspired by the structure and function of the
human brain. At their core, they consist of interconnected layers of artificial neurons,
each processing and transmitting signals.

A neural network has three fundamental layer types. The input layer receives raw data,
such as pixel values from an image or word embeddings from text. Hidden layers transform
this data through a series of mathematical operations. The output layer produces the
final prediction — a class label, a continuous value, or a probability distribution.

Each connection between neurons has a weight, a number that controls how strongly one
neuron influences another. During training, these weights are adjusted using backpropagation:
we compute how wrong the network's predictions are (the loss), then propagate this error
backward through the network, nudging each weight in the direction that reduces the error.

The learning rate is a critical hyperparameter. Set it too high, and the model overshoots
optimal weights and oscillates. Set it too low, and training takes forever and may get stuck
in local minima. Modern optimizers like Adam adaptively adjust learning rates per parameter.

Activation functions add non-linearity, which is essential — without them, stacking multiple
linear layers is mathematically equivalent to a single linear layer. ReLU (Rectified Linear
Unit) is the most popular: it outputs zero for negative inputs and the input itself for
positive inputs. Variants like Leaky ReLU and GELU address the "dying neuron" problem.

Deep networks with many hidden layers can learn hierarchical representations. In image
recognition, early layers detect edges, middle layers detect textures and shapes, and
final layers detect objects. This hierarchy is what makes deep learning so powerful for
complex tasks like image classification, natural language processing, and game playing.

Neural networks are notoriously data-hungry. A model with millions of parameters needs
proportionally large datasets to generalize. Techniques like dropout (randomly zeroing
neurons during training), L2 regularization, and data augmentation help prevent overfitting
on small datasets.
            """.strip(),
            metadata={"topic": "deep_learning", "difficulty": "beginner"}
        ),

        Document(
            doc_id="doc_002",
            title="Transformer Architecture and Attention Mechanism",
            text="""
The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need" by
Vaswani et al., revolutionized natural language processing and spawned the modern era of
large language models.

Before Transformers, sequence models used recurrence (RNNs, LSTMs) which processed tokens
one by one. This sequential nature prevented parallelization and caused the notorious
vanishing gradient problem for long sequences.

The key innovation of Transformers is the self-attention mechanism. For each token in a
sequence, self-attention computes a weighted sum of all other tokens, where the weights
represent relevance. This allows the model to directly relate "it" to "cat" in "The cat
sat on the mat and it was happy", regardless of distance.

Self-attention uses three learned projections: Queries (Q), Keys (K), and Values (V).
The computation is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

Where d_k is the key dimension. The sqrt(d_k) scaling prevents the dot products from
growing too large, which would push softmax into regions with tiny gradients.

Multi-head attention runs several attention operations in parallel with different learned
projections. Each head can attend to different aspects of the input: one head might track
syntactic relationships, another semantic roles.

A Transformer layer consists of: multi-head self-attention, followed by layer normalization
and a residual connection, followed by a feed-forward network (two linear layers with ReLU),
again followed by normalization and residual connection.

Positional encodings inject sequence order information, since attention is permutation-
invariant. The original paper used fixed sinusoidal encodings; modern models use learned
rotary position embeddings (RoPE) or ALiBi for better length generalization.

BERT uses bidirectional attention (seeing past and future context) for understanding tasks.
GPT uses causal attention (only seeing past context) for generation. The encoder-decoder
architecture (T5, BART) uses both: the encoder processes the input fully, while the decoder
attends to the encoder output and generates tokens autoregressively.
            """.strip(),
            metadata={"topic": "transformers", "difficulty": "intermediate"}
        ),

        Document(
            doc_id="doc_003",
            title="Large Language Models: GPT, Claude, and Gemini",
            text="""
Large Language Models (LLMs) are Transformer-based models trained on massive text corpora
with billions to trillions of tokens. They exhibit emergent capabilities not seen in smaller
models: reasoning, in-context learning, instruction following, and even rudimentary arithmetic.

GPT (Generative Pre-trained Transformer) by OpenAI uses a decoder-only architecture.
GPT-4, released in 2023, introduced multimodal capabilities (image + text). The GPT series
demonstrated that scaling laws hold: doubling parameters and data predictably improves
performance on most benchmarks.

Claude by Anthropic focuses on helpfulness, harmlessness, and honesty — the HHH framework.
Claude models use Constitutional AI (CAI): the model is trained with a set of principles
and learns to critique its own outputs. Claude 3 (Haiku, Sonnet, Opus) introduced a
family approach with different cost/capability tradeoffs.

Gemini by Google DeepMind was designed natively multimodal from the start. Unlike GPT-4
which added vision as an adapter, Gemini processes text, images, audio, and video with
a unified architecture. Gemini Ultra achieved human-expert performance on the MMLU benchmark.

LLaMA (Meta) democratized open-source LLMs. LLaMA 2 and 3 are freely available for research
and commercial use, enabling the open-source community to fine-tune and deploy powerful
models locally. Mistral, derived from LLaMA, uses grouped-query attention and sliding window
attention for efficiency.

RLHF (Reinforcement Learning from Human Feedback) is the key technique that turns a
raw pre-trained model into a useful assistant. Human raters compare model outputs; these
preferences train a reward model; the LLM is then fine-tuned with PPO to maximize reward.

Instruction tuning with supervised fine-tuning (SFT) on curated datasets is often sufficient
for many capabilities without full RLHF. Models like Alpaca and Vicuna showed instruction
tuning with ~50K examples dramatically improves instruction following.

Context windows have expanded dramatically: GPT-4 supports 128K tokens, Claude supports
200K tokens, and Gemini 1.5 Pro supports 1M tokens. This enables processing entire codebases
or books in a single prompt.
            """.strip(),
            metadata={"topic": "llms", "difficulty": "intermediate"}
        ),

        Document(
            doc_id="doc_004",
            title="Vector Databases and Similarity Search",
            text="""
Vector databases are purpose-built storage systems for high-dimensional vectors. They enable
fast approximate nearest-neighbor (ANN) search, which is the foundation of semantic search,
recommendation systems, and retrieval-augmented generation.

Traditional databases store structured data: integers, strings, dates. They excel at exact
matches (WHERE name = 'Alice') and range queries. But they cannot answer "find me documents
semantically similar to this query", which requires comparing floating-point vectors.

The core operation in vector search is: given a query vector q, find the k vectors in the
database most similar to q under some distance metric (cosine similarity, L2 distance,
dot product). Brute force requires O(N) comparisons per query — too slow at millions of docs.

Approximate Nearest Neighbor (ANN) algorithms trade perfect recall for speed:

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph. The top layers have
few nodes with long-range connections for fast navigation. Lower layers are denser for
precision. Queries start at the top and greedily navigate down. HNSW achieves sub-millisecond
queries with >95% recall at scale.

IVF (Inverted File Index, used by FAISS) clusters vectors using k-means. At query time,
only the nearest clusters are searched, dramatically reducing comparisons.

Qdrant is an open-source vector database written in Rust, known for:
- Rich payload filtering (filter by metadata while searching)
- On-disk indexing (handles datasets larger than RAM)
- Named vectors (multiple embedding models per document)
- Native sparse+dense hybrid search

Pinecone is a fully managed cloud vector database. It handles infrastructure automatically,
making it simple to deploy but with less control and higher cost.

pgvector is a PostgreSQL extension adding vector storage. If you already use Postgres, pgvector
lets you do similarity search without a separate service, at the cost of less optimization
than dedicated solutions.

Metadata filtering is crucial in practice. You might search for "machine learning papers"
but only within documents from 2023 AND where author = 'Hinton'. Vector databases support
this with pre-filtering (filter before search) or post-filtering (filter search results).
            """.strip(),
            metadata={"topic": "vector_databases", "difficulty": "intermediate"}
        ),

        Document(
            doc_id="doc_005",
            title="Retrieval-Augmented Generation (RAG) Systems",
            text="""
Retrieval-Augmented Generation (RAG) solves a fundamental limitation of large language models:
their knowledge is frozen at training time. RAG grounds model responses in fresh, specific
documents, dramatically reducing hallucination.

A RAG system has two phases: indexing and retrieval-generation.

In the indexing phase, documents are split into chunks (typically 256-512 tokens), each chunk
is converted to an embedding vector using a language model, and stored in a vector database
along with the original text.

In the retrieval-generation phase, the user's query is embedded with the same model, the
database returns the most similar chunks, these chunks are included in the LLM prompt as
context, and the LLM generates a grounded answer.

Advanced RAG techniques include:

Hypothetical Document Embeddings (HyDE): Instead of embedding the query directly, ask the
LLM to generate a hypothetical answer, then embed THAT as the query. Hypothetical answers
use vocabulary similar to actual documents, improving retrieval.

Query Decomposition: For complex questions ("Compare GPT-4 and Claude on coding"), decompose
into sub-queries ("GPT-4 coding performance", "Claude coding performance"), retrieve for each,
then synthesize.

Contextual Compression: After retrieving chunks, use an LLM to extract only the relevant
sentences, reducing noise before the generation step.

Reranking: Use a cross-encoder model to re-score retrieved chunks. Cross-encoders are slower
than bi-encoders but much more accurate as they process the query and document together.

The RAG evaluation framework RAGAS provides automated metrics: faithfulness (is the answer
grounded in the context?), answer relevancy (does the answer address the question?),
context precision (are retrieved chunks relevant?), and context recall (were all needed
facts retrieved?).

Common failure modes in RAG:
- Chunking too large: semantic signal diluted, retrieval imprecise
- Chunking too small: not enough context for generation
- Missing reranking: initial retrieval noise reaches the LLM
- No hybrid search: keyword queries (product names, codes) fail on pure dense retrieval
            """.strip(),
            metadata={"topic": "rag", "difficulty": "intermediate"}
        ),

        Document(
            doc_id="doc_006",
            title="BM25 and Sparse Retrieval Methods",
            text="""
BM25 (Best Match 25) is the gold standard for sparse lexical retrieval. It underlies search
engines like Elasticsearch, Apache Solr, and was the primary retrieval algorithm before
neural methods.

BM25 is an evolution of TF-IDF (Term Frequency — Inverse Document Frequency). TF-IDF scores
documents by how often query terms appear (TF) and how rare those terms are across the corpus
(IDF). BM25 improves on TF-IDF with two key modifications:

1. TF Saturation: In TF-IDF, a term appearing 100 times scores 100x higher than appearing once.
   BM25 saturates TF: the score approaches a ceiling as frequency increases. The parameter
   k1 (default 1.5) controls saturation speed.

2. Document Length Normalization: Long documents have more words, so terms appear more often
   by chance. BM25 normalizes scores by document length relative to corpus average. Parameter
   b (default 0.75) controls normalization strength.

The BM25 formula for term t in document D:
  score(t,D) = IDF(t) × [tf(t,D) × (k1+1)] / [tf(t,D) + k1×(1-b + b×|D|/avgdl)]

Where:
  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  N = number of documents, df(t) = documents containing term t
  |D| = length of document D, avgdl = average document length

BM25 shines for exact keyword matching. If a user searches for "BERT-large fine-tuning on
SQuAD 2.0", BM25 finds documents containing exactly those terms. Dense retrieval might return
semantically related but terminologically different documents.

The rank_bm25 Python library is the most popular implementation. For production, BM25 indices
are typically stored as inverted indices: a mapping from each term to the list of documents
containing it, with their TF scores. This enables O(|Q|) retrieval time (one lookup per
query term), extremely fast.

SPLADE (Sparse Lexical and Expansion model) is a neural evolution of sparse retrieval: it
uses a BERT model to generate sparse vectors (mostly zeros, a few non-zero entries) that
represent both the original terms AND semantically expanded terms. This bridges sparse and
dense worlds.
            """.strip(),
            metadata={"topic": "search_algorithms", "difficulty": "intermediate"}
        ),

        Document(
            doc_id="doc_007",
            title="Embedding Models and Sentence Transformers",
            text="""
Sentence Transformers (SBERT) are the backbone of modern semantic search and RAG systems.
They convert sentences and paragraphs into fixed-size dense vectors where semantic similarity
corresponds to geometric proximity.

SBERT (2019) solved a critical problem with original BERT: computing sentence similarity with
BERT required running the model on all (sentence1, sentence2) pairs — O(N²) complexity,
infeasible for large corpora. SBERT introduced the bi-encoder paradigm: embed each sentence
INDEPENDENTLY, then compare vectors. O(N) at index time, O(1) at query time.

Architecture: SBERT adds a pooling layer on top of BERT (mean pooling of all token embeddings
is standard, [CLS] pooling is also common). Training uses siamese and triplet networks with
contrastive loss: similar sentence pairs are pushed together, dissimilar pairs pushed apart.

The HuggingFace MTEB benchmark (Massive Text Embedding Benchmark) is the standard for
comparing embedding models. It covers 8 task categories:
- Retrieval: find relevant documents for queries
- STS (Semantic Textual Similarity): score sentence pair similarity
- Clustering: group similar documents
- Classification: embed then classify
- Reranking: sort documents by relevance
- Pair Classification: binary judgment of sentence pairs
- Summarization: embedding-based summary quality

Top models on MTEB (English):
- text-embedding-3-large (OpenAI): 64.6 avg score, 3072 dims, API only
- BAAI/bge-large-en-v1.5: 64.2 avg, 1024 dims, Apache 2.0
- thenlper/gte-large: 63.1 avg, 1024 dims, Apache 2.0
- BAAI/bge-small-en-v1.5: 62.2 avg, 384 dims — incredible efficiency
- all-MiniLM-L6-v2: 56.3 avg, 384 dims — the classic fast baseline

Matryoshka Representation Learning (MRL) trains models to produce vectors where the first
N dimensions are already meaningful. You can truncate a 1024-dim MRL vector to 128 dims
with minimal quality loss, saving storage and compute. OpenAI's text-embedding-3-* models
use MRL.

For multilingual retrieval, BAAI/bge-m3 and intfloat/multilingual-e5-large lead the MTEB
multilingual leaderboard. BGE-M3 uniquely outputs dense, sparse, AND multi-vector (ColBERT)
representations from a single model.
            """.strip(),
            metadata={"topic": "embeddings", "difficulty": "intermediate"}
        ),

        Document(
            doc_id="doc_008",
            title="Chunking Strategies for RAG",
            text="""
Chunking is the process of splitting documents into smaller pieces before indexing. The choice
of chunking strategy profoundly affects RAG quality. Chunks too large dilute semantic signal;
too small lack context for generation.

Fixed-Size Chunking splits text every N characters or tokens, optionally with overlap.
Overlap (typically 10-20% of chunk size) ensures that information at chunk boundaries is
captured. Simple, deterministic, fast. The main risk is splitting mid-sentence or mid-idea.

RecursiveCharacterTextSplitter (LangChain) is an improved fixed-size approach. It tries to
split on semantic boundaries in priority order: paragraph breaks (\n\n), then sentence breaks
(\n or ". "), then word boundaries, finally characters. This produces more coherent chunks
while still respecting size limits.

Semantic Chunking splits based on embedding similarity. The algorithm:
1. Split the document into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentences
4. When similarity drops below a threshold (topic shift), create a chunk boundary

This produces chunks with coherent topics, variable in size. More expensive (requires
embedding during indexing) but better semantic coherence.

Parent-Document Retrieval uses two granularities simultaneously. Index small chunks (128 tokens)
for precise retrieval, but return the parent chunk (512 tokens) for generation. The small
chunk's semantic signal is sharp; the large parent provides enough context for the LLM.
Implementation: store child chunks in vector DB with a parent_id field; retrieve child chunks,
then fetch parent documents using parent_ids.

Proposition Indexing (a recent technique): convert each chunk into atomic propositions — simple
factual statements. Index these propositions instead of raw chunks. Each proposition is
semantically precise, and retrieval recalls individual facts. Expensive (requires LLM at
index time) but powerful for fact-intensive Q&A.

Practical guidelines:
- Start with RecursiveCharacterTextSplitter at 512 tokens, 50-token overlap
- Upgrade to semantic chunking if retrieval quality suffers on topic-boundary questions
- Use parent-document for long-form generation that needs full paragraph context
- Evaluate chunk size empirically: typical sweet spot is 256-512 tokens for Q&A tasks
            """.strip(),
            metadata={"topic": "chunking", "difficulty": "beginner"}
        ),

        Document(
            doc_id="doc_009",
            title="Cross-Encoder Reranking for Improved Precision",
            text="""
Reranking is a two-stage retrieval pattern that combines the speed of ANN search with the
precision of cross-attention. It is now standard practice in production RAG systems.

Stage 1 (Bi-encoder retrieval) is fast because query and document are embedded independently.
The query embedding is compared to pre-computed document embeddings using dot product or
cosine similarity. This scales to millions of documents with sub-millisecond latency.

However, bi-encoders have a fundamental limitation: they compress each text into a single
vector, losing fine-grained interaction between query and document terms.

Stage 2 (Cross-encoder reranking) concatenates query + document as a single input:
[CLS] query [SEP] document [SEP]
A BERT-style model processes this joint input with full self-attention, allowing every
query token to attend to every document token. This captures subtle relevance signals
that bi-encoders miss.

Cross-encoders are 10-100x slower than bi-encoders (no pre-computation possible), making
them unsuitable for first-stage retrieval over large corpora. But applied to just 20-50
candidates from the first stage, latency is acceptable (50-200ms).

Popular cross-encoder models:
- cross-encoder/ms-marco-MiniLM-L-6-v2: Trained on MS MARCO, fast, good quality
- cross-encoder/ms-marco-electra-base: Slightly better quality, similar speed
- BAAI/bge-reranker-v2-m3: Best open-source, multilingual, supports long documents
- Cohere Rerank API: Best commercial option, extremely high quality

Practical reranking pattern:
1. Dense + BM25 retrieval: fetch top-50 candidates
2. Cross-encoder: score all 50 with (query, candidate) pairs
3. Sort by cross-encoder score
4. Return top-5 to LLM

The gain from reranking is largest when initial retrieval is noisy. In benchmarks, adding
reranking to a dense retriever typically improves NDCG@10 by 5-15 points.

ColBERT (Contextualized Late Interaction) is a middle ground: it stores one embedding per
token (not one per document), enabling richer matching than bi-encoders while being faster
than cross-encoders. MaxSim operation: for each query token, find the maximum similarity
with any document token, then sum across query tokens.
            """.strip(),
            metadata={"topic": "reranking", "difficulty": "advanced"}
        ),

        Document(
            doc_id="doc_010",
            title="RAGAS: Evaluating RAG Pipelines",
            text="""
RAGAS (RAG Assessment) is an evaluation framework that scores RAG pipelines on multiple
dimensions without requiring human-labeled data. It uses LLMs-as-judges to automate evaluation.

The key insight of RAGAS: evaluation traditionally requires human experts to judge answer
quality. RAGAS replaces humans with LLMs for certain metrics, enabling scalable automated
evaluation during development.

RAGAS Core Metrics:

1. Faithfulness measures whether the generated answer is factually consistent with the
retrieved context. RAGAS extracts all claims from the answer, then checks each claim
against the context using an LLM. Score = supported_claims / total_claims.
High faithfulness means low hallucination.

2. Answer Relevancy measures whether the answer addresses the original question. RAGAS
generates N hypothetical questions from the answer, then measures cosine similarity between
these questions and the original question. Irrelevant answers generate questions that differ
from the original.

3. Context Precision measures whether retrieved chunks are relevant to the question.
RAGAS uses an LLM to judge: "Given this question, is this retrieved chunk useful?"
Score = relevant_chunks_in_top_k / total_retrieved_chunks. High precision means no noise
in the context window.

4. Context Recall measures whether all information needed to answer is in the retrieved
context. Requires a ground truth answer. RAGAS checks what fraction of ground truth facts
appear in the retrieved context.

RAGAS in practice:
- Requires a test set: list of (question, ground_truth_answer) pairs
- Run your RAG pipeline on each question to get (context, answer)
- Pass (question, context, answer, ground_truth) to RAGAS evaluator
- Receives a DataFrame with metric scores per question + aggregated scores

Creating the test set:
- Manual: domain experts write Q&A pairs
- Semi-automatic: ask an LLM to generate questions from your documents
- RAGAS TestSet Generator: uses LLMs to generate diverse questions automatically

Interpreting scores:
- Faithfulness < 0.7: your LLM is hallucinating — improve context retrieval or prompting
- Answer Relevancy < 0.7: answer drifts from question — improve generation prompt
- Context Precision < 0.5: retriever fetches irrelevant chunks — improve retrieval
- Context Recall < 0.5: retriever misses relevant docs — check chunking and embedding model
            """.strip(),
            metadata={"topic": "evaluation", "difficulty": "intermediate"}
        ),
    ]

    return docs


if __name__ == "__main__":
    corpus = load_ai_corpus()
    print(f"Loaded {len(corpus)} documents:")
    for doc in corpus:
        word_count = len(doc.text.split())
        print(f"  {doc.doc_id}: '{doc.title}' ({word_count} words, topic={doc.metadata['topic']})")
