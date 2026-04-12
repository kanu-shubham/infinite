"""
bm25_retriever.py
=================
BM25 sparse retriever with detailed explanation of the algorithm.

THEORY RECAP (see THEORY.md Section 5):
  BM25 = Best Match 25, the gold standard for keyword search.
  It scores documents by term frequency (TF) and inverse document frequency (IDF)
  with saturation and length normalization.

  BM25 is what powers Elasticsearch, Apache Solr, and most search engines.
  For RAG, it's the "sparse" half of hybrid search.

FORMULA:
  score(D, Q) = Σ IDF(qᵢ) × [f(qᵢ,D)×(k1+1)] / [f(qᵢ,D) + k1×(1-b+b×|D|/avgdl)]

  IDF(qᵢ) = log((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) + 1)

  Parameters:
    k1 = 1.5  (TF saturation: higher = slower saturation)
    b  = 0.75 (length norm: 0=no norm, 1=full norm)
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Result from a retrieval operation."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict

    def __repr__(self):
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, chunk_id={self.chunk_id!r})"


class BM25Retriever:
    """
    BM25 sparse retriever using the rank_bm25 library.

    INDEXING TIME (offline):
      1. Tokenize all chunk texts
      2. Build inverted index (term → {doc: tf, ...})
      3. Compute IDF for each term
      4. Store BM25 object

    QUERY TIME (online):
      1. Tokenize query
      2. Look up each query term in inverted index
      3. Sum BM25 scores per document
      4. Return ranked results

    Time complexity: O(|Q| × avg_docs_per_term) — very fast

    TOKENIZATION NOTE:
      Simple whitespace + lowercase tokenization used here.
      Production: use stemming (Porter), stopword removal, or subword tokenization.
      For RAG over technical text, minimal tokenization often works well.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: TF saturation parameter. Higher = slower saturation.
                Typical range: 1.2-2.0
            b:  Document length normalization. 0=off, 1=full.
                Typical: 0.75
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = []  # store Chunk objects for result lookup
        self._is_indexed = False

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer: lowercase + split on non-alphanumeric characters.

        PRODUCTION NOTE:
            For better results, consider:
            - Removing stopwords ("the", "a", "is")
            - Stemming ("running" → "run")
            - Expanding synonyms
            But for learning, simple tokenization works fine.
        """
        import re
        # Lowercase and split on non-word characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def index(self, chunks) -> None:
        """
        Build the BM25 index from a list of Chunks.

        This is called ONCE at index time (offline), before any queries.

        Args:
            chunks: list of Chunk objects (from chunkers.py)

        WHAT HAPPENS INTERNALLY (rank_bm25 library):
            1. Tokenize all documents
            2. Compute document frequencies (df) for each term
            3. Compute average document length (avgdl)
            4. Store TF for each (document, term) pair
            → Ready to score any query instantly
        """
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._is_indexed = True
        print(f"BM25 index built: {len(self.chunks)} chunks, "
              f"vocab_size={len(self.bm25.idf)}")

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        Retrieve top-k chunks most relevant to the query by BM25 score.

        Args:
            query:  natural language or keyword query
            top_k:  number of results to return

        Returns:
            List of SearchResult sorted by score (highest first)

        EXAMPLE:
            Query: "attention mechanism transformer self-attention"
            BM25 looks for documents containing these exact words.
            "attention" is likely rare → high IDF → high score for docs with it.
            "the" would be common → low IDF → barely affects score.
        """
        if not self._is_indexed:
            raise RuntimeError("Call .index() before .search()")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # BM25 returns a score per document in the same order as the indexed corpus
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices by score (argsort in descending order)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue  # skip zero-score results
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=score,
                rank=rank + 1,
                metadata=chunk.metadata,
            ))

        return results

    def get_term_scores(self, query: str, text: str) -> Dict[str, float]:
        """
        Debug helper: show BM25 score contribution per query term for a given text.

        Useful for understanding WHY a document ranked where it did.
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(text)

        from rank_bm25 import BM25Okapi
        mini_bm25 = BM25Okapi([doc_tokens])

        term_scores = {}
        for token in query_tokens:
            score = mini_bm25.get_scores([token])[0]
            term_scores[token] = round(float(score), 4)

        return term_scores


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/infinite/rag")
    from corpus.corpus_loader import load_ai_corpus
    from corpus.chunkers import RecursiveCharacterChunker

    # Load and chunk corpus
    corpus = load_ai_corpus()
    chunker = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_corpus(corpus)

    # Build BM25 index
    retriever = BM25Retriever()
    retriever.index(chunks)

    # Test searches
    queries = [
        "self-attention mechanism transformer",
        "BM25 TF-IDF term frequency",
        "RAGAS faithfulness evaluation metrics",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retriever.search(query, top_k=3)
        for r in results:
            print(f"  Rank {r.rank} (score={r.score:.3f}): [{r.metadata.get('title', '?')}]")
            print(f"    '{r.text[:100]}...'")
