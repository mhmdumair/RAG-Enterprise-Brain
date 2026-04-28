"""
brain/chunker.py
================
Sentence-Aware Elastic Chunker.

Replaces the naive character sliding window with a semantically-aware
accumulation strategy that respects sentence boundaries.

Strategy:
    1. Split each block into sentences (NLTK preferred, regex fallback)
    2. Accumulate sentences until TARGET_SIZE (500 chars) is reached
    3. If next sentence would exceed HARD_LIMIT (800 chars):
       - Standard sentence  → migrate to next chunk (semantic bridge)
       - Oversized sentence → force-split at whitespace + is_fragment=True
    4. Overlap: last sentence carried into next chunk (contextual handshake)
    5. BBox: union of all source block bboxes in the chunk

Public interface is identical to the old chunker:
    chunker = TextChunker()
    chunks  = chunker.chunk(parsed_doc)  → list[TextChunk]

Nothing outside this file needs to change.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from core.config import settings
from core.logger import get_logger
from core.utils import make_chunk_id, normalize_bbox, NormalizedBBox
from brain.parser import ParsedDocument, TextBlock

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SIZE = 500   # soft target — accumulate until this many chars
HARD_LIMIT  = 800   # hard ceiling — never exceed
                    # 800 chars ≈ 200 tokens — safe within RoBERTa 512 limit

# ── Sentence splitter ─────────────────────────────────────────────────────────

# Simple fixed-width lookbehind — Python requires fixed width.
# Abbreviation false-splits are handled in post-processing (_split_sentences).
_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?])'   # preceded by sentence-ending punctuation
    r'(?!\d)'       # not followed by digit (avoids splitting "3.14")
    r'\s+',         # split on whitespace
    re.IGNORECASE,
)

# Abbreviations that should NOT trigger a sentence split
_ABBREVS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
    'fig', 'eq', 'no', 'vs', 'etc', 'approx',
    'e.g', 'i.e', 'al', 'pp', 'vol', 'dept', 'est',
}


def _split_sentences(text: str) -> list[str]:
    """
    Split a text block into individual sentences.

    Attempt order:
        1. NLTK sent_tokenize — most accurate for academic/technical text
        2. Regex split + abbreviation re-join — reliable fallback
        3. Return whole text — last resort if both fail

    Args:
        text — cleaned text from a single PyMuPDF block

    Returns:
        List of sentence strings, stripped, non-empty.
    """
    if not text.strip():
        return []

    # ── NLTK (preferred) ──────────────────────────────────────────────────────
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
        result = [s.strip() for s in sentences if s.strip()]
        if result:
            return result
    except Exception:
        pass

    # ── Regex + abbreviation fix (fallback) ───────────────────────────────────
    parts = _SENT_BOUNDARY.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [text.strip()]

    # Re-join fragments incorrectly split after abbreviations.
    # e.g. "Fig. 3 shows" must not be split after "Fig."
    merged: list[str] = []
    buffer = parts[0]

    for part in parts[1:]:
        last_word = buffer.rstrip().rsplit(None, 1)[-1].rstrip('.').lower()
        if last_word in _ABBREVS:
            buffer = buffer + " " + part
        else:
            merged.append(buffer)
            buffer = part

    if buffer:
        merged.append(buffer)

    return merged if merged else [text.strip()]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single semantically-coherent chunk ready for embedding.

    Interface is identical to the old TextChunk — all existing
    code that consumes TextChunk objects continues to work.

    Fields:
        chunk_id       — deterministic ID: sha256(doc_id:page:idx)[:16]
        document_id    — parent document reference
        filename       — source PDF filename
        page_number    — source page of the first sentence (1-based)
        chunk_index    — global position within the document (0-based)
        text           — complete sentences joined by spaces
        bbox           — union of all source block bboxes (normalized 0.0–1.0)
        char_start     — character start offset within source block
        char_end       — character end offset within source block
        is_fragment    — True if an oversized sentence was force-split
        sentence_count — number of complete sentences in this chunk
    """
    chunk_id:       str
    document_id:    str
    filename:       str
    page_number:    int
    chunk_index:    int
    text:           str
    bbox:           NormalizedBBox
    char_start:     int
    char_end:       int
    is_fragment:    bool = False
    sentence_count: int  = 0


# ── Internal buffer ───────────────────────────────────────────────────────────

@dataclass
class _ChunkBuffer:
    """
    Mutable accumulator used while building a single chunk.
    Internal to the chunker — never exposed outside this file.
    """
    sentences:     list[str]       = field(default_factory=list)
    source_blocks: list[TextBlock] = field(default_factory=list)
    char_start:    int             = 0
    is_fragment:   bool            = False

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def length(self) -> int:
        return len(self.text)

    def is_empty(self) -> bool:
        return len(self.sentences) == 0

    def last_sentence(self) -> Optional[str]:
        return self.sentences[-1] if self.sentences else None

    def add_sentence(self, sentence: str, block: TextBlock) -> None:
        self.sentences.append(sentence)
        if block not in self.source_blocks:
            self.source_blocks.append(block)

    def reset_keeping_overlap(
        self,
        overlap_sentence: Optional[str],
        block: TextBlock,
    ) -> None:
        """
        Reset buffer for next chunk, carrying the last sentence forward
        as the semantic bridge (contextual handshake).
        """
        self.sentences     = [overlap_sentence] if overlap_sentence else []
        self.source_blocks = [block]            if overlap_sentence else []
        self.char_start    = 0
        self.is_fragment   = False


# ── BBox union ────────────────────────────────────────────────────────────────

def _union_bboxes(blocks: list[TextBlock]) -> NormalizedBBox:
    """
    Compute the union bounding box across all source blocks in a chunk.

    The union expands to cover the outermost edges of all blocks so
    the PDF viewer highlights the entire logical paragraph.

    For single-block chunks this is equivalent to normalize_bbox().
    For multi-block chunks the highlight covers all contributing blocks.

    Args:
        blocks — list of TextBlock objects that contributed to this chunk

    Returns:
        NormalizedBBox with coordinates in 0.0–1.0 range.
    """
    if not blocks:
        logger.warning("_union_bboxes called with empty block list — using fallback")
        return NormalizedBBox(
            x0=0.0, y0=0.0, x1=1.0, y1=1.0,
            page_width=595.0, page_height=842.0,
        )

    page_width  = blocks[0].page_width
    page_height = blocks[0].page_height

    if page_width <= 0 or page_height <= 0:
        logger.warning(
            "Invalid page dimensions — clamping to 1.0",
            extra={
                "page_width":  page_width,
                "page_height": page_height,
            },
        )
        page_width  = max(page_width,  1.0)
        page_height = max(page_height, 1.0)

    raw_x0 = min(b.bbox["x0"] for b in blocks)
    raw_y0 = min(b.bbox["y0"] for b in blocks)
    raw_x1 = max(b.bbox["x1"] for b in blocks)
    raw_y1 = max(b.bbox["y1"] for b in blocks)

    return NormalizedBBox(
        x0=round(raw_x0 / page_width,  6),
        y0=round(raw_y0 / page_height, 6),
        x1=round(raw_x1 / page_width,  6),
        y1=round(raw_y1 / page_height, 6),
        page_width=page_width,
        page_height=page_height,
    )


# ── Main chunker ──────────────────────────────────────────────────────────────

class TextChunker:
    """
    Sentence-Aware Elastic Chunker.

    Public interface is identical to the previous character-based chunker.
    Drop-in replacement — nothing outside brain/chunker.py changes.

    Usage:
        chunker = TextChunker()
        chunks  = chunker.chunk(parsed_doc)
    """

    def __init__(
        self,
        target_size: int = TARGET_SIZE,
        hard_limit:  int = HARD_LIMIT,
    ):
        self._target = target_size
        self._limit  = hard_limit

        logger.info(
            "TextChunker (sentence-aware) initialized",
            extra={
                "target_size": self._target,
                "hard_limit":  self._limit,
            },
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        """
        Chunk all pages of a ParsedDocument into sentence-aware TextChunks.

        Args:
            doc — output of PDFParser.parse()

        Returns:
            Flat ordered list of TextChunk objects across all pages,
            ordered by page then position within the page.

        Signature is identical to the old chunker.
        brain/pipeline.py calls this and is completely unaffected.
        """
        all_chunks:  list[TextChunk] = []
        chunk_index: int             = 0

        for page in doc.pages:
            for block in page.blocks:
                block_chunks = self._chunk_block(
                    block=block,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    start_index=chunk_index,
                )
                all_chunks.extend(block_chunks)
                chunk_index += len(block_chunks)

        fragment_count = sum(1 for c in all_chunks if c.is_fragment)
        avg_sentences  = (
            round(
                sum(c.sentence_count for c in all_chunks) / len(all_chunks),
                1,
            )
            if all_chunks else 0
        )

        logger.info(
            "Document chunked",
            extra={
                "doc_file":      doc.filename,      # ← renamed from "filename"
                "total_chunks":  len(all_chunks),
                "pages":         len(doc.pages),
                "fragments":     fragment_count,
                "avg_sentences": avg_sentences,
            },
        )

        return all_chunks

    # ── Internal methods ──────────────────────────────────────────────────────

    def _chunk_block(
        self,
        block:       TextBlock,
        document_id: str,
        filename:    str,
        start_index: int,
    ) -> list[TextChunk]:
        """
        Apply elastic accumulation to one TextBlock.

        Steps:
            1. Split block text into sentences
            2. Accumulate sentences into buffer until TARGET_SIZE
            3. Flush buffer when HARD_LIMIT would be exceeded
            4. Carry last sentence into next chunk (semantic bridge)
            5. Handle oversized sentences with is_fragment flag

        Args:
            block       — single PyMuPDF text block
            document_id — parent document ID
            filename    — source PDF filename
            start_index — global chunk index at start of this block

        Returns:
            List of TextChunk objects for this block.
        """
        sentences = _split_sentences(block.text)
        if not sentences:
            return []

        chunks:      list[TextChunk] = []
        buffer:      _ChunkBuffer    = _ChunkBuffer()
        local_index: int             = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # ── Projected length if we add this sentence ──────────────────
            projected = (
                buffer.length + 1 + len(sentence)
                if not buffer.is_empty()
                else len(sentence)
            )

            # ── Hard limit would be exceeded — flush buffer first ─────────
            if projected > self._limit and not buffer.is_empty():
                chunk = self._finalize_chunk(
                    buffer=buffer,
                    block=block,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=start_index + local_index,
                )
                chunks.append(chunk)
                local_index += 1

                overlap = buffer.last_sentence()
                buffer.reset_keeping_overlap(overlap, block)

            # ── Oversized single sentence — force split ───────────────────
            if len(sentence) > self._limit:
                parts = self._force_split(sentence)
                for part in parts:
                    frag_buffer = _ChunkBuffer()
                    frag_buffer.add_sentence(part, block)
                    frag_buffer.is_fragment = True
                    chunk = self._finalize_chunk(
                        buffer=frag_buffer,
                        block=block,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=start_index + local_index,
                        is_fragment=True,
                    )
                    chunks.append(chunk)
                    local_index += 1
                buffer = _ChunkBuffer()
                continue

            # ── Normal accumulation ───────────────────────────────────────
            buffer.add_sentence(sentence, block)

            # ── Soft flush — at or past target size ───────────────────────
            if buffer.length >= self._target:
                chunk = self._finalize_chunk(
                    buffer=buffer,
                    block=block,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=start_index + local_index,
                )
                chunks.append(chunk)
                local_index += 1

                overlap = buffer.last_sentence()
                buffer.reset_keeping_overlap(overlap, block)

        # ── Flush remainder — last chunk of this block ────────────────────
        if not buffer.is_empty():
            chunk = self._finalize_chunk(
                buffer=buffer,
                block=block,
                document_id=document_id,
                filename=filename,
                chunk_index=start_index + local_index,
            )
            chunks.append(chunk)

        return chunks

    def _finalize_chunk(
        self,
        buffer:      _ChunkBuffer,
        block:       TextBlock,
        document_id: str,
        filename:    str,
        chunk_index: int,
        is_fragment: bool = False,
    ) -> TextChunk:
        """
        Convert a completed _ChunkBuffer into a TextChunk.

        Computes the union BBox across all source blocks and
        stamps the chunk with metadata.
        """
        text          = buffer.text
        source_blocks = buffer.source_blocks if buffer.source_blocks else [block]
        union_bbox    = _union_bboxes(source_blocks)

        return TextChunk(
            chunk_id=make_chunk_id(
                document_id,
                block.page_number,
                chunk_index,
            ),
            document_id=document_id,
            filename=filename,
            page_number=block.page_number,
            chunk_index=chunk_index,
            text=text,
            bbox=union_bbox,
            char_start=buffer.char_start,
            char_end=buffer.char_start + len(text),
            is_fragment=is_fragment or buffer.is_fragment,
            sentence_count=len(buffer.sentences),
        )

    def _force_split(self, sentence: str) -> list[str]:
        """
        Split a single oversized sentence at whitespace boundaries.

        Only triggered when one sentence exceeds HARD_LIMIT.
        Rare in academic/technical documents.

        Args:
            sentence — the oversized sentence string

        Returns:
            List of sub-strings each under HARD_LIMIT characters.
        """
        parts:   list[str] = []
        current: str       = ""

        for word in sentence.split():
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) > self._limit and current:
                parts.append(current.strip())
                current = word
            else:
                current = candidate

        if current:
            parts.append(current.strip())

        logger.warning(
            "Oversized sentence force-split",
            extra={
                "original_length": len(sentence),
                "parts":           len(parts),
            },
        )

        return parts