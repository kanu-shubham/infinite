"""
bm25.py — Keyword-based retrieval using BM25 (Best Match 25).

BM25 is the algorithm behind Elasticsearch and Lucene.
Unlike TF-IDF it handles two extra problems:
  1. Term Saturation: the 100th mention of "python" should not count
     100x more than the 1st mention. BM25 caps the benefit.
  2. Length normalisation: a short doc that says "python" once is more
     relevant than a long doc that says "python" once buried in 10k words.

Formula for a single query term t in document d:
    BM25(t, d) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))

    where:
        tf   = term frequency in d
        k1   = 1.5  (saturation constant — controls how fast score plateaus)
        b    = 0.75 (length penalty — 0 = no penalty, 1 = full normalisation)
        dl   = length of d in tokens
        avgdl= average document length across corpus
"""
from rank_bm25 import BM25Okapi
from documents import Chunk


class BM25Index:
    """
    Wraps rank_bm25.BM25Okapi to work with our Chunk objects.

    Internal state:
        _chunks   : list[Chunk] in insertion order
        _bm25     : BM25Okapi object (built from tokenised corpus)
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Add chunks and (re)build the BM25 index.
        BM25Okapi must be built all at once — it needs avgdl across the full corpus.
        """
        self._chunks.extend(chunks)
        self._rebuild()

    def remove_chunks(self, chunk_ids: set[str]) -> None:
        """Remove chunks and rebuild."""
        self._chunks = [c for c in self._chunks if c.chunk_id not in chunk_ids]
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        # Tokenise each chunk: split on whitespace, lowercase.
        # In production you'd use a proper tokeniser (spaCy, NLTK).
        tokenised_corpus = [c.content.lower().split() for c in self._chunks]
        # BM25Okapi precomputes IDF and avgdl from the whole corpus.
        self._bm25 = BM25Okapi(tokenised_corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """
        Returns [(chunk, bm25_score), ...] sorted by score descending.

        BM25 scores are NOT cosine similarities — they have no fixed range.
        A score of 8.0 vs 3.0 just means "more relevant", not a probability.
        """
        if self._bm25 is None:
            return []

        query_tokens = query.lower().split()

        # get_scores returns a NumPy array of length N (one score per chunk)
        scores = self._bm25.get_scores(query_tokens)

        # pair each score with its chunk, sort descending, take top_k
        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
