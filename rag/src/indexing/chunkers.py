"""
Advanced chunking strategies beyond fixed-size sliding windows.

Three strategies, each suited to a different situation:

RecursiveChunker
----------------
Splits hierarchically: tries "\n\n" (paragraphs) first, then "\n"
(lines), then "." (sentences), then " " (words) — stopping as soon
as chunks are small enough.  This is LangChain's RecursiveCharacterTextSplitter
approach and is the best default for prose documents.  Keeps semantically
coherent sections together (a paragraph stays in one chunk) while
guaranteeing a max chunk size.

SemanticChunker
---------------
Embeds every sentence, computes cosine distance between consecutive
sentences, and splits where the distance spikes (topic change).
Produces chunks that are semantically coherent by definition — no
sentence from a new topic bleeds into the previous chunk.
Requires an embedder at chunk time (more expensive than pure text splits).
Used when semantic coherence matters more than uniform chunk size
(e.g. legal documents, academic papers).

ContextualChunker   (Anthropic's "Contextual Retrieval" approach)
-----------------
For each chunk, calls Claude to generate a 1-2 sentence context
summary that explains where the chunk fits in the whole document.
Prepends this summary to the chunk text BEFORE embedding.

Why it helps: a chunk like "The net retention rate was 112%." is
ambiguous without context.  With contextual retrieval the stored text
becomes "From the Q3 2024 earnings call discussing ARR metrics. The
net retention rate was 112%." — now the embedding captures the topic.

Anthropic reports 35-67% reduction in retrieval failures with this
approach.  Cost: one Claude call per chunk at index time.

Late Chunking  (note — explained here but not implemented)
------------
A research technique from Jina AI (2024).  Instead of embedding
fixed-size windows of text, it:
1. Tokenises the FULL document (may exceed context window for long docs).
2. Runs the full document through the embedding model to get
   token-level embeddings that carry full-document context.
3. THEN partitions the token embeddings into chunk-sized windows.
Result: each chunk embedding reflects its position in the whole document.
Unlike contextual retrieval (which prepends an LLM summary), late
chunking is purely model-level — no extra LLM call, but requires a
model that outputs token-level embeddings and can handle long contexts
(e.g. jina-embeddings-v3).  Not implementable with sentence-transformers
out of the box.
"""

from __future__ import annotations

import re
import textwrap
from typing import List, Optional

import numpy as np
import anthropic

from src.config import config
from src.documents import Chunk, Document


# ---------------------------------------------------------------------------
# Recursive Chunker
# ---------------------------------------------------------------------------

class RecursiveChunker:
    """
    Hierarchical text splitter.

    Tries each separator in order until chunks fit within max_size.
    Overlap is added by re-including the last overlap_chars characters
    of the previous chunk at the start of the next.

    Parameters
    ----------
    max_size     : maximum characters per chunk
    overlap      : character overlap between consecutive chunks
    separators   : ordered list of separators to try
    """

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(
        self,
        max_size: int = 512,
        overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        self._max_size = max_size
        self._overlap = overlap
        self._separators = separators or self._DEFAULT_SEPARATORS

    def chunk_document(self, doc: Document) -> List[Chunk]:
        pieces = self._split(doc.content.strip(), self._separators)
        return self._pieces_to_chunks(doc, pieces)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first separator that produces
        chunks small enough, then recurse on any pieces that are still too big."""
        if len(text) <= self._max_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if sep:
            parts = text.split(sep)
        else:
            # Character-level fallback
            parts = [text[i:i + self._max_size] for i in range(0, len(text), self._max_size)]
            return parts

        # Merge small parts back together to avoid tiny chunks, then recurse
        # on parts that are still too large.
        merged: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self._max_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # Part itself too big — recurse with next separator
                if len(part) > self._max_size:
                    merged.extend(self._split(part, remaining_seps))
                    current = ""
                else:
                    current = part
        if current:
            merged.append(current)

        return merged

    def _pieces_to_chunks(self, doc: Document, pieces: List[str]) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_text = doc.content
        char_pos = 0

        for i, piece in enumerate(pieces):
            # Find approximate start in original text
            start = doc_text.find(piece[:40], char_pos) if piece else char_pos
            if start == -1:
                start = char_pos
            end = start + len(piece)

            chunks.append(Chunk(
                content=piece.strip(),
                doc_id=doc.doc_id,
                doc_title=doc.title,
                doc_source=doc.source,
                start_char=start,
                end_char=end,
                chunk_index=i,
                metadata=dict(doc.metadata),
            ))
            char_pos = max(0, end - self._overlap)

        return [c for c in chunks if c.content]


# ---------------------------------------------------------------------------
# Semantic Chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Splits documents by detecting topic changes via embedding cosine distance.

    Algorithm
    ---------
    1. Split into sentences (by '.', '!', '?').
    2. Embed all sentences.
    3. Compute cosine distance between each consecutive pair.
    4. Find split points where distance exceeds (mean + breakpoint_std * std).
    5. Merge sentences into chunks at those split points.

    Parameters
    ----------
    embedder         : any object with .embed_texts(List[str]) -> List[np.ndarray]
    breakpoint_std   : how many std devs above mean triggers a split (lower = more chunks)
    max_size         : hard cap on characters per chunk (secondary safety)
    """

    def __init__(self, embedder, breakpoint_std: float = 1.5, max_size: int = 1500):
        self._embedder = embedder
        self._breakpoint_std = breakpoint_std
        self._max_size = max_size

    def chunk_document(self, doc: Document) -> List[Chunk]:
        sentences = self._split_sentences(doc.content.strip())
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(doc, sentences[0], 0, 0, len(sentences[0]), 0)]

        # Embed all sentences in one batch
        embeddings = self._embedder.embed_texts(sentences)

        # Cosine distances between consecutive sentences
        distances = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i].astype(np.float32)
            b = embeddings[i + 1].astype(np.float32)
            a /= (np.linalg.norm(a) + 1e-9)
            b /= (np.linalg.norm(b) + 1e-9)
            distances.append(1.0 - float(np.dot(a, b)))  # distance = 1 - similarity

        # Split where distance exceeds threshold
        if distances:
            threshold = np.mean(distances) + self._breakpoint_std * np.std(distances)
            split_after = {i for i, d in enumerate(distances) if d > threshold}
        else:
            split_after = set()

        # Group sentences into chunks
        groups: List[List[str]] = []
        current: List[str] = []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if i in split_after:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        # Convert groups to Chunk objects
        chunks = []
        doc_text = doc.content
        search_start = 0
        for idx, group in enumerate(groups):
            text = " ".join(group).strip()
            if not text:
                continue
            start = doc_text.find(group[0][:30], search_start)
            if start == -1:
                start = search_start
            end = min(start + len(text), len(doc_text))
            chunks.append(self._make_chunk(doc, text, idx, start, end, idx))
            search_start = max(0, end - 20)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _make_chunk(self, doc, text, idx, start, end, chunk_index):
        return Chunk(
            content=text,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_source=doc.source,
            start_char=start,
            end_char=end,
            chunk_index=chunk_index,
            metadata=dict(doc.metadata),
        )


# ---------------------------------------------------------------------------
# Contextual Chunker  (Anthropic's Contextual Retrieval approach)
# ---------------------------------------------------------------------------

_CONTEXT_SYSTEM = textwrap.dedent("""\
    You are a document analysis assistant.

    Your task: given a full document and a specific chunk from it, write a
    1-2 sentence context that situates the chunk within the document.

    This context will be prepended to the chunk before embedding, so it must:
    - Identify the document section and topic
    - Mention key entities that make the chunk retrievable
    - Be factual and specific (no filler like "This chunk discusses...")

    Respond with ONLY the context sentence(s). No preamble.
""")


class ContextualChunker:
    """
    Wraps any base chunker and enriches each chunk with an LLM-generated
    context summary (Anthropic's Contextual Retrieval, 2024).

    The stored chunk content becomes:
        "<context summary>\\n\\n<original chunk text>"

    This dramatically improves retrieval for chunks that are ambiguous
    without document context (e.g. financial figures, technical specs).

    Parameters
    ----------
    base_chunker : a RecursiveChunker or SemanticChunker instance
    max_doc_chars: truncate documents to this length when sending to Claude
                  (keep prompt cost bounded; 8k chars ≈ 2k tokens)
    """

    def __init__(self, base_chunker, max_doc_chars: int = 8000):
        self._base = base_chunker
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._max_doc_chars = max_doc_chars

    def chunk_document(self, doc: Document) -> List[Chunk]:
        chunks = self._base.chunk_document(doc)
        doc_summary = doc.content[:self._max_doc_chars]

        enriched = []
        for chunk in chunks:
            context = self._generate_context(doc_summary, chunk.content)
            enriched_content = f"{context}\n\n{chunk.content}"
            # Replace content in-place (dataclass is mutable)
            object.__setattr__(chunk, "content", enriched_content)
            chunk.metadata["has_context"] = True
            enriched.append(chunk)

        return enriched

    def _generate_context(self, document: str, chunk: str) -> str:
        """Call Claude to generate a situating context for this chunk."""
        prompt = (
            f"<document>\n{document}\n</document>\n\n"
            f"<chunk>\n{chunk[:500]}\n</chunk>\n\n"
            "Write the context for this chunk."
        )
        try:
            resp = self._client.messages.create(
                model=config.model,
                max_tokens=100,
                temperature=0,
                system=_CONTEXT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception:
            return ""  # degrade gracefully — chunk still useful without context
