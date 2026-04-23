"""
brain/chunker.py
================
Sliding-window text chunker.

Takes ParsedDocument output from brain/parser.py and splits
each text block into overlapping chunks suitable for embedding.

Why overlap?
    If an answer spans the boundary between two chunks,
    overlap ensures at least one chunk captures the full context.

Flow:
    ParsedDocument → per-block sliding window → list[TextChunk]
"""

from dataclasses import dataclass
from core.config import settings
from core.logger import get_logger
from core.utils import make_chunk_id, normalize_bbox, NormalizedBBox
from brain.parser import ParsedDocument, TextBlock

logger = get_logger(__name__)


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """
    A single chunk of text ready for embedding.

    Fields:
        chunk_id     — deterministic ID: sha256(doc_id:page:idx)[:16]
        document_id  — parent document reference
        filename     — source PDF filename
        page_number  — source page (1-based)
        chunk_index  — global chunk index within the document (0-based)
        text         — the chunk text (400 chars target, 80 overlap)
        bbox         — normalized BBox (0.0–1.0) of the source block
        char_start   — character start offset within the source block text
        char_end     — character end offset within the source block text
    """
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    bbox: NormalizedBBox
    char_start: int
    char_end: int


# ── Chunker ───────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits a ParsedDocument into overlapping TextChunks.

    Strategy:
        For each text block on each page, apply a sliding window
        of size CHUNK_SIZE with CHUNK_OVERLAP step back.
        If a block is shorter than CHUNK_SIZE, it becomes one chunk.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk(parsed_doc)
    """

    def __init__(self):
        self._chunk_size = settings.chunk_size
        self._overlap = settings.chunk_overlap
        self._step = self._chunk_size - self._overlap
        logger.info(
            "TextChunker initialized",
            extra={
                "chunk_size": self._chunk_size,
                "overlap": self._overlap,
                "step": self._step,
            },
        )

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        """
        Chunk all pages of a ParsedDocument.

        Args:
            doc — output of PDFParser.parse()

        Returns:
            Flat list of TextChunk objects across all pages,
            ordered by page then position.
        """
        all_chunks: list[TextChunk] = []
        global_chunk_index = 0

        for page in doc.pages:
            for block in page.blocks:
                block_chunks = self._chunk_block(
                    block=block,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    start_index=global_chunk_index,
                )
                all_chunks.extend(block_chunks)
                global_chunk_index += len(block_chunks)

        logger.info(
            "Document chunked",
            extra={
                "doc_filename": doc.filename,  # Changed from 'filename'
                "total_chunks": len(all_chunks),
                "pages": len(doc.pages),
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
        Apply sliding window to a single TextBlock.

        If the block text is shorter than chunk_size,
        it becomes a single chunk with no splitting.
        """
        text = block.text
        text_len = len(text)
        chunks: list[TextChunk] = []

        # Normalize the block BBox once — reused for all sub-chunks
        norm_bbox = normalize_bbox(
            bbox=block.bbox,
            page_width=block.page_width,
            page_height=block.page_height,
        )

        # Single chunk — block fits within chunk_size
        if text_len <= self._chunk_size:
            chunk = self._make_chunk(
                text=text,
                char_start=0,
                char_end=text_len,
                chunk_index=start_index,
                document_id=document_id,
                filename=filename,
                page_number=block.page_number,
                bbox=norm_bbox,
            )
            return [chunk]

        # Sliding window — block is larger than chunk_size
        local_index = 0
        pos = 0
        while pos < text_len:
            end = min(pos + self._chunk_size, text_len)
            chunk_text = text[pos:end]

            # Don't create a tiny trailing chunk — merge with previous
            if len(chunk_text) < self._overlap and chunks:
                break

            chunk = self._make_chunk(
                text=chunk_text,
                char_start=pos,
                char_end=end,
                chunk_index=start_index + local_index,
                document_id=document_id,
                filename=filename,
                page_number=block.page_number,
                bbox=norm_bbox,
            )
            chunks.append(chunk)
            local_index += 1

            if end == text_len:
                break
            pos += self._step  # slide forward by (chunk_size - overlap)

        return chunks

    def _make_chunk(
        self,
        text: str,
        char_start: int,
        char_end: int,
        chunk_index: int,
        document_id: str,
        filename: str,
        page_number: int,
        bbox: NormalizedBBox,
    ) -> TextChunk:
        return TextChunk(
            chunk_id=make_chunk_id(document_id, page_number, chunk_index),
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            chunk_index=chunk_index,
            text=text,
            bbox=bbox,
            char_start=char_start,
            char_end=char_end,
        )