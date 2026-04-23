"""
brain/parser.py
===============
PDF text and bounding box extractor using PyMuPDF (fitz).

For each page it returns a list of TextBlock objects, each containing:
  - the raw text
  - pixel-space BBox (x0, y0, x1, y1)
  - page number (1-based)
  - page dimensions (width, height) for BBox normalization

Flow:
    PDF file → fitz.open() → per-page text blocks → list[ParsedPage]
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from pathlib import Path

from core.config import settings
from core.logger import get_logger
from core.exceptions import PDFParseError, FileTooLargeError
from core.utils import clean_text, RawBBox

logger = get_logger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """
    A single block of text extracted from a PDF page.
    One page can have many blocks (paragraphs, headers, table cells, etc.)

    Fields:
        text         — cleaned text content of the block
        bbox         — pixel-space bounding box from PyMuPDF
        page_number  — 1-based page number
        block_index  — position of this block on its page (0-based)
        page_width   — page width in points (for BBox normalization)
        page_height  — page height in points (for BBox normalization)
    """
    text: str
    bbox: RawBBox
    page_number: int
    block_index: int
    page_width: float
    page_height: float


@dataclass
class ParsedPage:
    """
    All text blocks extracted from a single PDF page.

    Fields:
        page_number  — 1-based page number
        blocks       — ordered list of TextBlock objects
        page_width   — page width in points
        page_height  — page height in points
    """
    page_number: int
    blocks: list[TextBlock] = field(default_factory=list)
    page_width: float = 0.0
    page_height: float = 0.0

    @property
    def full_text(self) -> str:
        """Return all block texts joined as a single string."""
        return " ".join(b.text for b in self.blocks if b.text)


@dataclass
class ParsedDocument:
    """
    Complete parsed output for one PDF file.

    Fields:
        filename     — original filename
        document_id  — deterministic ID (set by pipeline)
        pages        — list of ParsedPage objects
        total_pages  — total pages in the original PDF
        file_size_mb — file size at parse time
    """
    filename: str
    document_id: str
    pages: list[ParsedPage] = field(default_factory=list)
    total_pages: int = 0
    file_size_mb: float = 0.0

    @property
    def all_blocks(self) -> list[TextBlock]:
        """Flat list of all TextBlocks across all pages."""
        return [block for page in self.pages for block in page.blocks]


# ── Parser ────────────────────────────────────────────────────────────────────

class PDFParser:
    """
    Extracts text blocks and BBox coordinates from a PDF file.

    Usage:
        parser = PDFParser()
        doc = parser.parse(Path("document.pdf"), document_id="abc123")
    """

    def __init__(self):
        self._max_pages = settings.max_pages_per_pdf
        self._max_size_mb = settings.max_file_size_mb
        logger.info(
            "PDFParser initialized",
            extra={
                "max_pages": self._max_pages,
                "max_size_mb": self._max_size_mb,
            },
        )

    def parse(self, file_path: Path, document_id: str) -> ParsedDocument:
        """
        Parse a PDF file and return a ParsedDocument.

        Args:
            file_path    — absolute path to the PDF file
            document_id  — pre-computed document ID (from make_document_id)

        Returns:
            ParsedDocument with all pages and text blocks populated.

        Raises:
            FileTooLargeError  — if file exceeds max_file_size_mb
            PDFParseError      — if PyMuPDF cannot open or read the file
        """
        file_path = Path(file_path)
        filename = file_path.name

        # ── Size check ────────────────────────────────────────────────────────
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self._max_size_mb:
            raise FileTooLargeError(filename, size_mb, self._max_size_mb)

        logger.info(
            "Parsing PDF",
            extra={"pdf_filename": filename, "size_mb": round(size_mb, 2)},  # Changed from 'filename'
        )

        # ── Open PDF ──────────────────────────────────────────────────────────
        try:
            pdf = fitz.open(str(file_path))
        except Exception as exc:
            raise PDFParseError(filename, f"Cannot open file: {exc}") from exc

        parsed_doc = ParsedDocument(
            filename=filename,
            document_id=document_id,
            total_pages=len(pdf),
            file_size_mb=round(size_mb, 3),
        )

        # ── Process pages ─────────────────────────────────────────────────────
        pages_to_process = min(len(pdf), self._max_pages)

        for page_idx in range(pages_to_process):
            try:
                parsed_page = self._parse_page(pdf[page_idx], page_idx + 1)
                if parsed_page.blocks:  # skip completely empty pages
                    parsed_doc.pages.append(parsed_page)
            except Exception as exc:
                logger.warning(
                    "Skipping page due to error",
                    extra={"pdf_filename": filename, "page": page_idx + 1, "error": str(exc)},  # Changed from 'filename'
                )
                continue

        pdf.close()

        if not parsed_doc.pages:
            raise PDFParseError(filename, "No extractable text found in document.")

        logger.info(
            "PDF parsed successfully",
            extra={
                "pdf_filename": filename,  # Changed from 'filename'
                "pages_processed": len(parsed_doc.pages),
                "total_blocks": len(parsed_doc.all_blocks),
            },
        )

        return parsed_doc

    def _parse_page(self, page: fitz.Page, page_number: int) -> ParsedPage:
        """
        Extract all text blocks from a single fitz.Page.

        PyMuPDF's get_text("blocks") returns tuples of:
            (x0, y0, x1, y1, text, block_no, block_type)
        block_type 0 = text, 1 = image — we skip images.
        """
        rect = page.rect
        parsed_page = ParsedPage(
            page_number=page_number,
            page_width=rect.width,
            page_height=rect.height,
        )

        raw_blocks = page.get_text("blocks")

        for block_index, block in enumerate(raw_blocks):
            x0, y0, x1, y1, raw_text, _block_no, block_type = block

            # Skip image blocks
            if block_type != 0:
                continue

            text = clean_text(raw_text)
            if not text or len(text) < 10:  # skip noise (page numbers, artifacts)
                continue

            text_block = TextBlock(
                text=text,
                bbox=RawBBox(x0=x0, y0=y0, x1=x1, y1=y1),
                page_number=page_number,
                block_index=block_index,
                page_width=rect.width,
                page_height=rect.height,
            )
            parsed_page.blocks.append(text_block)

        return parsed_page