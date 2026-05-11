"""
brain/chunker.py
================
Sentence-Aware Elastic Chunker with Sparse Linked Chunks.

Changes:
    - TextChunk now has next_chunk_id and prev_chunk_id fields
    - Links created ONLY when a sentence spans chunk boundaries
    - No links for chunks that fit completely within limits
    - Links replace duplication (semantic bridge) with efficient pointers
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

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

_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?])'   # preceded by sentence-ending punctuation
    r'(?!\d)'       # not followed by digit (avoids splitting "3.14")
    r'\s+',         # split on whitespace
    re.IGNORECASE,
)

_ABBREVS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
    'fig', 'eq', 'no', 'vs', 'etc', 'approx',
    'e.g', 'i.e', 'al', 'pp', 'vol', 'dept', 'est',
}


def _split_sentences(text: str) -> list[str]:
    """Split a text block into individual sentences."""
    if not text.strip():
        return []

    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
        result = [s.strip() for s in sentences if s.strip()]
        if result:
            return result
    except Exception:
        pass

    parts = _SENT_BOUNDARY.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [text.strip()]

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

    Fields:
        chunk_id       — deterministic ID
        document_id    — parent document reference
        filename       — source PDF filename
        page_number    — source page (1-based)
        chunk_index    — global position (0-based)
        text           — complete sentences joined by spaces
        bbox           — union of all source block bboxes
        char_start     — character start offset
        char_end       — character end offset
        is_fragment    — True if oversized sentence was force-split
        sentence_count — number of complete sentences
        next_chunk_id  — ID of next chunk in sequence (if sentence spans)
        prev_chunk_id  — ID of previous chunk in sequence (if sentence spans)
        is_linked      — True if this chunk is part of a link chain
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
    sentence_count: int = 0
    next_chunk_id:  Optional[str] = None
    prev_chunk_id:  Optional[str] = None
    is_linked:      bool = False


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
    will_have_next: bool = False  # True if next chunk will continue this sentence

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

    def reset(self) -> None:
        """Clear buffer completely (no overlap)."""
        self.sentences = []
        self.source_blocks = []
        self.char_start = 0
        self.is_fragment = False
        self.will_have_next = False


# ── BBox union ────────────────────────────────────────────────────────────────

def _union_bboxes(blocks: list[TextBlock]) -> NormalizedBBox:
    """Compute the union bounding box across all source blocks."""
    if not blocks:
        logger.warning("_union_bboxes called with empty block list — using fallback")
        return NormalizedBBox(
            x0=0.0, y0=0.0, x1=1.0, y1=1.0,
            page_width=595.0, page_height=842.0,
        )

    page_width = blocks[0].page_width
    page_height = blocks[0].page_height

    if page_width <= 0 or page_height <= 0:
        page_width = max(page_width, 1.0)
        page_height = max(page_height, 1.0)

    raw_x0 = min(b.bbox["x0"] for b in blocks)
    raw_y0 = min(b.bbox["y0"] for b in blocks)
    raw_x1 = max(b.bbox["x1"] for b in blocks)
    raw_y1 = max(b.bbox["y1"] for b in blocks)

    return NormalizedBBox(
        x0=round(raw_x0 / page_width, 6),
        y0=round(raw_y0 / page_height, 6),
        x1=round(raw_x1 / page_width, 6),
        y1=round(raw_y1 / page_height, 6),
        page_width=page_width,
        page_height=page_height,
    )


# ── Main chunker ──────────────────────────────────────────────────────────────

class TextChunker:
    """
    Sentence-Aware Elastic Chunker with Sparse Linked Chunks.

    Links are created ONLY when a sentence naturally spans chunk boundaries.
    Most chunks remain independent (no links).
    """

    def __init__(
        self,
        target_size: int = TARGET_SIZE,
        hard_limit: int = HARD_LIMIT,
    ):
        self._target = target_size
        self._limit = hard_limit

        logger.info(
            "TextChunker (sentence-aware + sparse links) initialized",
            extra={
                "target_size": self._target,
                "hard_limit": self._limit,
            },
        )

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        """Chunk all pages of a ParsedDocument into sentence-aware TextChunks."""
        all_chunks: list[TextChunk] = []
        chunk_index: int = 0

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

        # Post-process: Add links between chunks from the same block
        all_chunks = self._add_links_between_chunks(all_chunks)

        fragment_count = sum(1 for c in all_chunks if c.is_fragment)
        linked_count = sum(1 for c in all_chunks if c.is_linked)
        avg_sentences = (
            round(sum(c.sentence_count for c in all_chunks) / len(all_chunks), 1)
            if all_chunks else 0
        )

        logger.info(
            "Document chunked",
            extra={
                "doc_file": doc.filename,
                "total_chunks": len(all_chunks),
                "pages": len(doc.pages),
                "fragments": fragment_count,
                "linked_chunks": linked_count,
                "avg_sentences": avg_sentences,
            },
        )

        return all_chunks

    def _chunk_block(
        self,
        block: TextBlock,
        document_id: str,
        filename: str,
        start_index: int,
    ) -> list[TextChunk]:
        """
        Apply elastic accumulation to one TextBlock.
        Returns chunks WITHOUT links (links added in post-processing).
        """
        sentences = _split_sentences(block.text)
        if not sentences:
            return []

        chunks: list[TextChunk] = []
        buffer: _ChunkBuffer = _ChunkBuffer()
        local_index: int = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            projected = (
                buffer.length + 1 + len(sentence)
                if not buffer.is_empty()
                else len(sentence)
            )

            # ── Hard limit would be exceeded — flush buffer ────────────────
            if projected > self._limit and not buffer.is_empty():
                # This chunk will have a next chunk (sentence continues)
                buffer.will_have_next = True
                
                chunk = self._finalize_chunk(
                    buffer=buffer,
                    block=block,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=start_index + local_index,
                )
                chunks.append(chunk)
                local_index += 1

                # Start new chunk with the overflowing sentence
                buffer.reset()
                buffer.add_sentence(sentence, block)
                continue

            # ── Oversized single sentence — force split ───────────────────
            if len(sentence) > self._limit:
                if not buffer.is_empty():
                    chunk = self._finalize_chunk(
                        buffer=buffer,
                        block=block,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=start_index + local_index,
                    )
                    chunks.append(chunk)
                    local_index += 1
                    buffer.reset()

                parts = self._force_split(sentence)
                for j, part in enumerate(parts):
                    is_first = (j == 0)
                    is_last = (j == len(parts) - 1)
                    frag_buffer = _ChunkBuffer()
                    frag_buffer.add_sentence(part, block)
                    frag_buffer.is_fragment = True
                    frag_buffer.will_have_next = not is_last
                    
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
                buffer.reset()

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

    def _add_links_between_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """
        Add prev/next links between chunks that are part of a sentence chain.
        Only chunks that were created with will_have_next get links.
        """
        if len(chunks) <= 1:
            return chunks

        for i, chunk in enumerate(chunks):
            # Check if this chunk expects a next chunk
            # Also check if next chunk naturally continues (same page, close index)
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                
                # Only link if chunks are from the same page and close in index
                # (This indicates they came from the same sentence chain)
                if (chunk.page_number == next_chunk.page_number and
                    next_chunk.chunk_index == chunk.chunk_index + 1):
                    
                    # Check if chunk ends mid-sentence (doesn't end with .!?)
                    if not self._ends_with_sentence_boundary(chunk.text):
                        chunk.next_chunk_id = next_chunk.chunk_id
                        chunk.is_linked = True
                        next_chunk.prev_chunk_id = chunk.chunk_id
                        next_chunk.is_linked = True

        return chunks

    def _ends_with_sentence_boundary(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        text = text.strip()
        if not text:
            return True
        return text[-1] in '.!?'

    def _finalize_chunk(
        self,
        buffer: _ChunkBuffer,
        block: TextBlock,
        document_id: str,
        filename: str,
        chunk_index: int,
        is_fragment: bool = False,
    ) -> TextChunk:
        """Convert a completed _ChunkBuffer into a TextChunk."""
        text = buffer.text
        source_blocks = buffer.source_blocks if buffer.source_blocks else [block]
        union_bbox = _union_bboxes(source_blocks)

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
            next_chunk_id=None,
            prev_chunk_id=None,
            is_linked=False,
        )

    def _force_split(self, sentence: str) -> list[str]:
        """Split a single oversized sentence at whitespace boundaries."""
        parts: list[str] = []
        current: str = ""

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
                "parts": len(parts),
            },
        )

        return parts