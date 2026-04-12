"""
ragas_eval.py
=============
RAGAS-style evaluation of RAG pipelines.

THEORY RECAP (see THEORY.md Section 7):
  RAGAS evaluates 4 dimensions:
    1. Faithfulness      — is the answer grounded in the context?
    2. Answer Relevancy  — does the answer address the question?
    3. Context Precision — are retrieved chunks relevant?
    4. Context Recall    — were all needed facts retrieved?

  Real RAGAS uses an LLM (GPT-4, Claude) as the judge.
  Here we implement BOTH:
    A) A self-contained version using cosine similarity (no LLM needed)
    B) Integration with the real RAGAS library (requires LLM API key)

  The self-contained version teaches you HOW each metric works.
  The real RAGAS version is what you'd use in production.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """
    A single evaluation sample for RAG.

    question:     The user's question
    contexts:     List of retrieved text chunks (what the LLM saw)
    answer:       The LLM's generated answer
    ground_truth: The correct answer (needed for context_recall)
    """
    question: str
    contexts: List[str]
    answer: str
    ground_truth: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalScores:
    """Scores for one evaluation sample."""
    faithfulness: float = 0.0        # 0–1: is answer grounded in context?
    answer_relevancy: float = 0.0    # 0–1: does answer address the question?
    context_precision: float = 0.0   # 0–1: are retrieved chunks relevant?
    context_recall: float = 0.0      # 0–1: were all needed facts retrieved?

    @property
    def ragas_score(self) -> float:
        """
        Harmonic mean of all 4 metrics (RAGAS aggregate score).

        WHY HARMONIC MEAN?
            Harmonic mean penalizes low values more than arithmetic mean.
            A pipeline scoring (0.9, 0.9, 0.9, 0.1) should NOT be considered
            good — the 0.1 recall is a critical failure.
            Harmonic mean: 4 / (1/0.9 + 1/0.9 + 1/0.9 + 1/0.1) ≈ 0.31
            Arithmetic mean: (0.9+0.9+0.9+0.1)/4 = 0.70 (misleadingly high)
        """
        scores = [self.faithfulness, self.answer_relevancy,
                  self.context_precision, self.context_recall]
        if any(s == 0 for s in scores):
            return 0.0
        return len(scores) / sum(1/s for s in scores)

    def __repr__(self):
        return (f"EvalScores(faith={self.faithfulness:.3f}, "
                f"relevancy={self.answer_relevancy:.3f}, "
                f"precision={self.context_precision:.3f}, "
                f"recall={self.context_recall:.3f}, "
                f"ragas={self.ragas_score:.3f})")


class SelfContainedRAGAS:
    """
    RAGAS metrics implemented WITHOUT requiring an LLM API key.

    Uses embedding-based similarity as a proxy for LLM judgment.
    This teaches you exactly how each metric works.

    ACCURACY NOTE:
        LLM-based RAGAS is more accurate (can understand paraphrasing,
        logical entailment, etc.). Embedding-based is a reasonable proxy
        and is 100% free and offline.
    """

    def __init__(self, embedder):
        """
        Args:
            embedder: DenseEmbedder instance for computing similarities
        """
        self.embedder = embedder

    # ── Metric 1: Faithfulness ─────────────────────────────────────────────

    def faithfulness(self, sample: EvalSample) -> float:
        """
        Are all claims in the answer supported by the context?

        REAL RAGAS METHOD (with LLM):
            1. Use LLM to extract all factual claims from the answer
            2. For each claim, ask LLM: "Is this claim supported by the context?"
            3. faithfulness = supported_claims / total_claims

        OUR PROXY METHOD (without LLM):
            1. Split answer into sentences
            2. For each sentence, compute cosine similarity with context
            3. A sentence is "supported" if max similarity with any context > threshold
            4. faithfulness = supported_sentences / total_sentences

        WHY THIS WORKS:
            If the answer sentence is semantically close to something in the context,
            it's likely grounded in the context. If it's very different, it may be
            hallucinated.

        LIMITATION:
            Can't detect logical contradictions, only semantic dissimilarity.
        """
        if not sample.answer or not sample.contexts:
            return 0.0

        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', sample.answer)
                     if len(s.strip()) > 20]
        if not sentences:
            return 1.0

        # Embed answer sentences and context
        answer_vecs = self.embedder.embed_documents(sentences)
        context_vecs = self.embedder.embed_documents(sample.contexts)

        # For each answer sentence, find max similarity with any context chunk
        import numpy as np
        # Shape: (num_sentences, num_contexts)
        sim_matrix = answer_vecs @ context_vecs.T

        supported_count = 0
        threshold = 0.4  # sentence is "supported" if sim > this

        for i, sentence in enumerate(sentences):
            max_sim = float(np.max(sim_matrix[i]))
            if max_sim >= threshold:
                supported_count += 1

        return supported_count / len(sentences)

    # ── Metric 2: Answer Relevancy ─────────────────────────────────────────

    def answer_relevancy(self, sample: EvalSample) -> float:
        """
        Does the answer address the question that was asked?

        REAL RAGAS METHOD:
            1. Use LLM to generate N hypothetical questions FROM the answer
               (e.g., N=3: "What does the answer describe?")
            2. Embed original question and the N generated questions
            3. relevancy = avg cosine_sim(original_question, generated_question_i)

            INSIGHT: If the answer is relevant, questions derived from it should
            be similar to the original question. If the answer drifts off-topic,
            the generated questions will be about different topics.

        OUR PROXY:
            1. Embed the question
            2. Embed the answer
            3. relevancy = cosine_sim(question_vec, answer_vec)

            Limitation: doesn't penalize overly verbose/off-topic answers as well.
        """
        if not sample.answer or not sample.question:
            return 0.0

        q_vec = self.embedder.embed_query(sample.question)
        a_vec = self.embedder.embed_documents([sample.answer])[0]

        return float(q_vec @ a_vec)

    # ── Metric 3: Context Precision ────────────────────────────────────────

    def context_precision(self, sample: EvalSample) -> float:
        """
        Of the retrieved chunks, how many are actually relevant to the question?

        REAL RAGAS METHOD:
            For each context chunk (in retrieval rank order):
              Ask LLM: "Is this chunk relevant to answer: {question}?"
              precision@k = (relevant chunks in top-k) / k
              context_precision = avg precision@k for k=1,...,K (Average Precision)

        OUR PROXY:
            For each context chunk, compute cosine_sim(question, chunk).
            A chunk is "relevant" if sim > threshold.
            precision = relevant_chunks / total_chunks

        IMPORTANT: This metric penalizes retrieving irrelevant noise.
        A retriever that fetches 10 chunks, 3 of which are off-topic, scores 0.7.
        """
        if not sample.contexts or not sample.question:
            return 0.0

        q_vec = self.embedder.embed_query(sample.question)
        ctx_vecs = self.embedder.embed_documents(sample.contexts)

        import numpy as np
        similarities = ctx_vecs @ q_vec  # cosine similarity for each context

        threshold = 0.25
        relevant_count = int(np.sum(similarities >= threshold))
        return relevant_count / len(sample.contexts)

    # ── Metric 4: Context Recall ───────────────────────────────────────────

    def context_recall(self, sample: EvalSample) -> float:
        """
        Were all the facts needed to answer the question in the retrieved context?

        REAL RAGAS METHOD (requires ground_truth):
            1. Use LLM to extract all facts/claims from ground_truth answer
            2. For each fact, ask LLM: "Is this fact present in the context?"
            3. recall = supported_facts / total_facts

        OUR PROXY:
            Split ground_truth into sentences.
            For each sentence, compute max cosine_sim with any context chunk.
            A fact is "present" if max similarity > threshold.
            recall = present_facts / total_facts

        REQUIRES: ground_truth to be set in EvalSample.
        """
        if not sample.ground_truth or not sample.contexts:
            return 0.0

        gt_sentences = [s.strip() for s in re.split(r'[.!?]', sample.ground_truth)
                        if len(s.strip()) > 20]
        if not gt_sentences:
            return 0.0

        import numpy as np
        gt_vecs = self.embedder.embed_documents(gt_sentences)
        ctx_vecs = self.embedder.embed_documents(sample.contexts)

        # For each GT fact, find max similarity with any context chunk
        sim_matrix = gt_vecs @ ctx_vecs.T  # (num_facts, num_contexts)
        max_sims = np.max(sim_matrix, axis=1)

        threshold = 0.4
        supported = int(np.sum(max_sims >= threshold))
        return supported / len(gt_sentences)

    def evaluate(self, sample: EvalSample) -> EvalScores:
        """Compute all 4 metrics for a single sample."""
        return EvalScores(
            faithfulness=self.faithfulness(sample),
            answer_relevancy=self.answer_relevancy(sample),
            context_precision=self.context_precision(sample),
            context_recall=self.context_recall(sample),
        )

    def evaluate_dataset(self, samples: List[EvalSample]) -> Dict:
        """
        Evaluate a list of samples and return aggregate statistics.

        Returns a dict with:
            per_sample: list of {question, scores, EvalScores}
            aggregate: {mean_faithfulness, mean_relevancy, ..., mean_ragas}
        """
        all_scores = []
        per_sample = []

        for sample in samples:
            scores = self.evaluate(sample)
            all_scores.append(scores)
            per_sample.append({
                "question": sample.question,
                "answer": sample.answer[:100] + "..." if len(sample.answer) > 100 else sample.answer,
                "faithfulness": scores.faithfulness,
                "answer_relevancy": scores.answer_relevancy,
                "context_precision": scores.context_precision,
                "context_recall": scores.context_recall,
                "ragas_score": scores.ragas_score,
            })

        n = len(all_scores)
        if n == 0:
            return {"error": "No samples"}

        aggregate = {
            "num_samples": n,
            "faithfulness": round(sum(s.faithfulness for s in all_scores) / n, 4),
            "answer_relevancy": round(sum(s.answer_relevancy for s in all_scores) / n, 4),
            "context_precision": round(sum(s.context_precision for s in all_scores) / n, 4),
            "context_recall": round(sum(s.context_recall for s in all_scores) / n, 4),
            "ragas_score": round(sum(s.ragas_score for s in all_scores) / n, 4),
        }

        return {
            "aggregate": aggregate,
            "per_sample": per_sample,
        }


def build_test_set() -> List[Dict]:
    """
    A test set of Q&A pairs matching our synthetic corpus.

    In production you'd either:
    1. Manually write these (most accurate but slow)
    2. Use RAGAS TestSet Generator (auto-generates from your docs)
    3. Use domain experts

    Each entry: {"question": ..., "ground_truth": ...}
    """
    return [
        {
            "question": "What is the key innovation of the Transformer architecture?",
            "ground_truth": "The key innovation is the self-attention mechanism, which allows the model to directly relate any two tokens in a sequence regardless of their distance, unlike RNNs which process tokens sequentially."
        },
        {
            "question": "What are the parameters k1 and b in BM25?",
            "ground_truth": "k1 is the TF saturation parameter controlling how fast term frequency saturates, typically 1.2-2.0. b is the document length normalization parameter, where 0 means no normalization and 1 means full normalization, typically set to 0.75."
        },
        {
            "question": "What is RAGAS and what metrics does it use?",
            "ground_truth": "RAGAS is an evaluation framework for RAG pipelines that uses LLMs as judges. Its four metrics are faithfulness (answer grounded in context), answer relevancy (answer addresses the question), context precision (retrieved chunks are relevant), and context recall (all needed facts were retrieved)."
        },
        {
            "question": "What is HNSW and why is it used in vector databases?",
            "ground_truth": "HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor algorithm. It builds a multi-layer graph with long-range connections in top layers and dense connections in bottom layers, enabling sub-millisecond queries with over 95% recall at scale."
        },
        {
            "question": "What is parent-document retrieval in RAG?",
            "ground_truth": "Parent-document retrieval is a two-level chunking strategy where small child chunks are indexed for precise retrieval, but when a child chunk matches, the larger parent document is returned to the LLM. This provides precise retrieval signal (small chunks) with rich generation context (large parents)."
        },
        {
            "question": "Which embedding models are top-ranked on MTEB for English?",
            "ground_truth": "Top English embedding models on MTEB include BAAI/bge-large-en-v1.5 (64.2 score, 1024 dims), thenlper/gte-large (63.1), and BAAI/bge-small-en-v1.5 (62.2, 384 dims). OpenAI text-embedding-3-large leads overall at 64.6 but is API-only."
        },
        {
            "question": "What is Reciprocal Rank Fusion and when is it used?",
            "ground_truth": "Reciprocal Rank Fusion (RRF) is a method to combine multiple ranked lists. The score for a document is the sum of 1/(k + rank) across all lists, where k=60 is a constant. It is used in hybrid search to combine BM25 and dense retrieval results because their scores have incompatible scales."
        },
        {
            "question": "How does a cross-encoder differ from a bi-encoder?",
            "ground_truth": "A bi-encoder embeds query and document independently then compares vectors, enabling pre-computation but losing fine-grained interaction. A cross-encoder processes the concatenated query and document together with full self-attention, seeing both simultaneously, which is much more accurate but cannot pre-compute document representations."
        },
    ]


def print_eval_report(eval_results: Dict) -> None:
    """Pretty-print evaluation results."""
    agg = eval_results["aggregate"]
    per = eval_results["per_sample"]

    print("\n" + "="*70)
    print("RAGAS EVALUATION REPORT")
    print("="*70)
    print(f"\nDataset: {agg['num_samples']} questions\n")

    print("AGGREGATE SCORES:")
    print(f"  Faithfulness      (↑ = less hallucination):  {agg['faithfulness']:.4f}")
    print(f"  Answer Relevancy  (↑ = on-topic answers):    {agg['answer_relevancy']:.4f}")
    print(f"  Context Precision (↑ = less retrieval noise): {agg['context_precision']:.4f}")
    print(f"  Context Recall    (↑ = covers all facts):    {agg['context_recall']:.4f}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  RAGAS Score       (harmonic mean):           {agg['ragas_score']:.4f}")

    print(f"\nPER-QUESTION BREAKDOWN:")
    print(f"{'#':<4} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'RAGAS':>7}  Question")
    print(f"{'─'*70}")
    for i, s in enumerate(per, 1):
        print(f"{i:<4} {s['faithfulness']:>7.3f} {s['answer_relevancy']:>7.3f} "
              f"{s['context_precision']:>7.3f} {s['context_recall']:>7.3f} "
              f"{s['ragas_score']:>7.3f}  {s['question'][:45]}...")

    # Diagnosis
    print(f"\nDIAGNOSIS:")
    issues = []
    if agg['faithfulness'] < 0.7:
        issues.append("Low faithfulness → LLM generating info not in context; try better retrieval or stricter prompt")
    if agg['answer_relevancy'] < 0.7:
        issues.append("Low relevancy → answers drifting off-topic; improve generation prompt")
    if agg['context_precision'] < 0.5:
        issues.append("Low precision → retrieving irrelevant chunks; improve retrieval or add reranker")
    if agg['context_recall'] < 0.5:
        issues.append("Low recall → missing relevant documents; check chunking size and embedding model")
    if not issues:
        issues.append("All metrics look healthy!")
    for issue in issues:
        print(f"  ⚠  {issue}")
    print("="*70)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/infinite/rag")
    from embeddings.embedder import DenseEmbedder

    embedder = DenseEmbedder()
    evaluator = SelfContainedRAGAS(embedder)

    # Simulate a RAG pipeline output for demonstration
    samples = [
        EvalSample(
            question="What is self-attention?",
            contexts=[
                "The key innovation of Transformers is the self-attention mechanism. "
                "For each token in a sequence, self-attention computes a weighted sum of all other tokens.",
                "Attention uses three learned projections: Queries (Q), Keys (K), and Values (V).",
            ],
            answer="Self-attention is a mechanism where each token attends to all other tokens in the sequence, computing weighted sums using Query, Key, and Value projections.",
            ground_truth="Self-attention allows each token to directly relate to any other token through Q, K, V projections and softmax weighting."
        ),
        EvalSample(
            question="What is BM25?",
            contexts=[
                "BM25 is an evolution of TF-IDF with TF saturation (k1 parameter) and document length normalization (b parameter).",
                "BM25 shines for exact keyword matching, used by Elasticsearch and Apache Solr.",
            ],
            answer="BM25 is a keyword search algorithm with TF saturation and document length normalization, widely used in search engines.",
            ground_truth="BM25 is a scoring function for document retrieval using term frequency saturation and inverse document frequency, with parameters k1 and b."
        ),
    ]

    results = evaluator.evaluate_dataset(samples)
    print_eval_report(results)
