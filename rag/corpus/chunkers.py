"""
chunkers.py
===========
Implements three chunking strategies with detailed comments explaining each.

THEORY RECAP (see THEORY.md Section 3 for full theory):
  Chunking = splitting docs into pieces that fit in an embedding model's context.
  Strategy choice affects retrieval precision dramatically.

Strategies implemented:
  1. RecursiveCharacterChunker  — simple, fast, production-ready baseline
  2. SemanticChunker            — topic-aware, uses embeddings
  3. ParentDocumentChunker      — two-level: precise child + context-rich parent
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from corpus.corpus_loader import Document


@dataclass
class Chunk:
    """
    A chunk is a piece of a document ready for embedding and indexing.

    Attributes:
        chunk_id    : unique id, e.g. "doc_001_chunk_0"
        doc_id      : which document this came from
        text        : the actual text content
        char_start  : character offset in original document
        char_end    : character offset in original document
        parent_id   : for parent-doc retrieval, the id of the parent chunk
        metadata    : inherited from parent doc plus chunk-level info
    """
    chunk_id: str
    doc_id: str
    text: str
    char_start: int = 0
    char_end: int = 0
    parent_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Chunk(id={self.chunk_id!r}, words={len(self.text.split())}, doc={self.doc_id!r})"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Recursive Character Chunker
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveCharacterChunker:
    """
    Splits text by trying a list of separators in priority order.

    THEORY:
        Plain fixed-size splitting cuts at arbitrary character positions.
        Recursive splitting tries smarter splits first:
          \n\n  → paragraph breaks (best: each chunk = one idea)
          \n    → line breaks
          ". "  → sentence ends
          " "   → word boundaries
          ""    → characters (last resort)

        With overlap, the last `overlap` characters of one chunk repeat at
        the start of the next. This ensures context at boundaries isn't lost.

    WHEN TO USE:
        - Default starting point for any RAG project
        - Fast (no model calls at index time)
        - Deterministic (reproducible splits)
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        Args:
            chunk_size:    max characters per chunk (≈ chunk_size / 4 tokens)
            chunk_overlap: characters to repeat from prev chunk (context preservation)
            separators:    try these in order; falls back to next if text too large
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the first separator that produces
        pieces small enough, then recurse on any oversized pieces.
        """
        # Pick the first separator that exists in the text
        separator = separators[-1]  # fallback: char-by-char
        for sep in separators:
            if sep == "" or sep in text:
                separator = sep
                break

        # Split on the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Collect splits that need further splitting and those that don't
        good_splits: List[str] = []
        current_splits: List[str] = []

        for s in splits:
            if len(s) < self.chunk_size:
                current_splits.append(s)
            else:
                # This split is still too large → recurse with next separator
                if current_splits:
                    good_splits.extend(self._merge_splits(current_splits, separator))
                    current_splits = []
                # Recurse with remaining separators
                remaining = separators[separators.index(separator) + 1:] if separator in separators[:-1] else [""]
                good_splits.extend(self._split_text(s, remaining))

        if current_splits:
            good_splits.extend(self._merge_splits(current_splits, separator))

        return good_splits

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge splits into chunks respecting chunk_size with overlap.

        Example:
            splits = ["Hello world.", "My name is Alice.", "I like cats."]
            chunk_size = 30, overlap = 10

            chunk1 = "Hello world. My name is Alice."  (28 chars, fits)
            chunk2 = " Alice. I like cats."            (overlap from prev)
        """
        chunks: List[str] = []
        current_doc: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len + len(separator) > self.chunk_size and current_doc:
                # Flush current chunk
                if current_doc:
                    chunk_text = separator.join(current_doc)
                    chunks.append(chunk_text)

                    # Build overlap: keep last pieces that fit in overlap window
                    while current_doc and current_len > self.chunk_overlap:
                        removed = current_doc.pop(0)
                        current_len -= len(removed) + len(separator)

            current_doc.append(split)
            current_len += split_len + len(separator)

        if current_doc:
            chunks.append(separator.join(current_doc))

        return [c.strip() for c in chunks if c.strip()]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split a single Document into Chunks."""
        raw_splits = self._split_text(doc.text, self.separators)
        chunks = []
        char_pos = 0

        for i, text in enumerate(raw_splits):
            chunk_id = f"{doc.doc_id}_rc_{i:03d}"
            start = doc.text.find(text, char_pos)
            if start == -1:
                start = char_pos
            end = start + len(text)
            char_pos = max(0, end - self.chunk_overlap)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                char_start=start,
                char_end=end,
                metadata={
                    **doc.metadata,
                    "title": doc.title,
                    "source": doc.source,
                    "chunk_strategy": "recursive",
                    "chunk_index": i,
                }
            ))

        return chunks

    def chunk_corpus(self, docs: List[Document]) -> List[Chunk]:
        """Chunk all documents in a corpus."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. Semantic Chunker
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Splits documents based on semantic similarity between consecutive sentences.

    THEORY:
        When two consecutive sentences have LOW cosine similarity, it signals
        a topic shift — a natural chunk boundary.

        Algorithm:
          1. Split into sentences
          2. Embed each sentence
          3. Compute cosine similarity between s[i] and s[i+1]
          4. Find "breakpoints": positions where similarity < threshold
          5. Form chunks between breakpoints

    WHEN TO USE:
        - When documents cover multiple topics
        - When fixed-size chunks cut across topic boundaries
        - Accept: slower indexing, variable chunk sizes, requires model

    KEY PARAMETER:
        breakpoint_threshold: lower = more splits (more, smaller chunks)
                              higher = fewer splits (fewer, larger chunks)
                              default 0.3 works well for most content
    """

    def __init__(
        self,
        embed_fn=None,
        breakpoint_threshold: float = 0.3,
        min_chunk_chars: int = 100,
    ):
        """
        Args:
            embed_fn:              function(list[str]) → list of vectors (numpy arrays)
                                   If None, falls back to recursive chunking.
            breakpoint_threshold:  cosine sim below this → new chunk
            min_chunk_chars:       don't create chunks smaller than this
        """
        self.embed_fn = embed_fn
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_chars = min_chunk_chars

    def _cosine_similarity(self, a, b) -> float:
        """Cosine similarity between two vectors."""
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter using regex."""
        # Split on ". ", "! ", "? " but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split using semantic similarity if embed_fn is available."""
        if self.embed_fn is None:
            # Fallback to recursive chunking without embeddings
            fallback = RecursiveCharacterChunker()
            return fallback.chunk_document(doc)

        sentences = self._split_into_sentences(doc.text)
        if len(sentences) <= 2:
            # Too short to split semantically
            return [Chunk(
                chunk_id=f"{doc.doc_id}_sem_000",
                doc_id=doc.doc_id,
                text=doc.text,
                metadata={**doc.metadata, "title": doc.title, "chunk_strategy": "semantic"}
            )]

        # Embed all sentences at once (batched for efficiency)
        embeddings = self.embed_fn(sentences)

        # Compute pairwise cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints: where similarity drops below threshold
        breakpoints = set()
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                breakpoints.add(i + 1)  # split BEFORE sentence i+1

        # Build chunks from sentence groups
        chunks = []
        current_sentences = [sentences[0]]

        for i, sentence in enumerate(sentences[1:], 1):
            if i in breakpoints:
                chunk_text = " ".join(current_sentences)
                if len(chunk_text) >= self.min_chunk_chars:
                    chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_sem_{len(chunks):03d}",
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        metadata={
                            **doc.metadata,
                            "title": doc.title,
                            "source": doc.source,
                            "chunk_strategy": "semantic",
                            "chunk_index": len(chunks),
                        }
                    ))
                    current_sentences = [sentence]
                else:
                    # Too small, keep accumulating
                    current_sentences.append(sentence)
            else:
                current_sentences.append(sentence)

        # Don't forget the last group
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_sem_{len(chunks):03d}",
                doc_id=doc.doc_id,
                text=chunk_text,
                metadata={
                    **doc.metadata,
                    "title": doc.title,
                    "source": doc.source,
                    "chunk_strategy": "semantic",
                    "chunk_index": len(chunks),
                }
            ))

        return chunks

    def chunk_corpus(self, docs: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parent-Document Chunker
# ─────────────────────────────────────────────────────────────────────────────

class ParentDocumentChunker:
    """
    Two-level chunking: small child chunks for retrieval, large parent for generation.

    THEORY:
        Tension in RAG:
          - For retrieval: small chunks → precise semantic signal
          - For generation: large chunks → enough context for the LLM

        Parent-document retrieval resolves this:
          1. Create large parent chunks (e.g., 512 chars)
          2. Split each parent into small child chunks (e.g., 128 chars)
          3. Index ONLY child chunks in vector DB
          4. Each child has a parent_id pointing to its parent
          5. At retrieval: find child chunks, return their parents to LLM

        This way, retrieval is precise (small chunk semantics) but the LLM
        gets rich context (parent chunk).

    DATA STRUCTURE:
        parent_store: dict[parent_id → Chunk]  (for lookup at retrieval time)
        children:     list[Chunk]               (indexed in vector DB)

    WHEN TO USE:
        - Q&A on structured documents (papers, manuals, legal text)
        - When generation needs paragraph-level context
        - When retrieval precision is low with large chunks
    """

    def __init__(
        self,
        parent_chunk_size: int = 600,
        child_chunk_size: int = 150,
        parent_overlap: int = 50,
        child_overlap: int = 20,
    ):
        self.parent_chunker = RecursiveCharacterChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
        )
        self.child_chunker = RecursiveCharacterChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
        )

    def chunk_document(self, doc: Document) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Returns (parent_chunks, child_chunks).

        Each child_chunk.parent_id points to its parent_chunk.chunk_id.
        Child chunks are what you index in the vector DB.
        Parent chunks are stored separately for retrieval-time lookup.
        """
        parent_chunks = self.parent_chunker.chunk_document(doc)

        # Rename parent chunks to use 'parent' in ID
        for i, pc in enumerate(parent_chunks):
            pc.chunk_id = f"{doc.doc_id}_parent_{i:03d}"
            pc.metadata["chunk_strategy"] = "parent_document"
            pc.metadata["is_parent"] = True

        # Create child chunks from each parent
        all_children = []
        for parent in parent_chunks:
            # Create a temporary Document from parent text for child chunking
            parent_doc = Document(
                doc_id=parent.chunk_id,
                title=doc.title,
                text=parent.text,
                metadata=parent.metadata,
            )
            children = self.child_chunker.chunk_document(parent_doc)

            for j, child in enumerate(children):
                child.chunk_id = f"{parent.chunk_id}_child_{j:03d}"
                child.doc_id = doc.doc_id  # original doc id
                child.parent_id = parent.chunk_id
                child.metadata["chunk_strategy"] = "parent_document"
                child.metadata["is_parent"] = False
                child.metadata["parent_id"] = parent.chunk_id

            all_children.extend(children)

        return parent_chunks, all_children

    def chunk_corpus(self, docs: List[Document]) -> Tuple[List[Chunk], List[Chunk]]:
        """Returns (all_parents, all_children) across the corpus."""
        all_parents, all_children = [], []
        for doc in docs:
            parents, children = self.chunk_document(doc)
            all_parents.extend(parents)
            all_children.extend(children)
        return all_parents, all_children


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/user/infinite/rag")

    from corpus.corpus_loader import load_ai_corpus

    corpus = load_ai_corpus()
    doc = corpus[0]  # "Introduction to Neural Networks"

    print("=" * 60)
    print("STRATEGY 1: Recursive Character Chunker")
    print("=" * 60)
    chunker1 = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=40)
    chunks1 = chunker1.chunk_document(doc)
    print(f"Document: '{doc.title}' ({len(doc.text)} chars)")
    print(f"Produced {len(chunks1)} chunks:")
    for c in chunks1:
        print(f"  {c.chunk_id}: {len(c.text)} chars | '{c.text[:60]}...'")

    print()
    print("=" * 60)
    print("STRATEGY 3: Parent-Document Chunker")
    print("=" * 60)
    chunker3 = ParentDocumentChunker(parent_chunk_size=600, child_chunk_size=150)
    parents, children = chunker3.chunk_document(doc)
    print(f"Produced {len(parents)} parents and {len(children)} children")
    for p in parents[:2]:
        p_children = [c for c in children if c.parent_id == p.chunk_id]
        print(f"  Parent {p.chunk_id} ({len(p.text)} chars) → {len(p_children)} children")
        for c in p_children:
            print(f"    Child {c.chunk_id} ({len(c.text)} chars): '{c.text[:50]}...'")
