"""
tests/test_parser.py
====================
Tests for brain/parser.py

Covers:
  - Successful parse of a valid PDF
  - Text block extraction with BBox coordinates
  - Page number is 1-based
  - Page limit enforcement (max_pages_per_pdf)
  - File too large raises FileTooLargeError
  - Non-existent file raises PDFParseError
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from brain.parser import PDFParser, ParsedDocument, TextBlock
from core.exceptions import FileTooLargeError, PDFParseError
from core.utils import make_document_id

# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def get_test_pdf() -> Path:
    """Return the first PDF found in fixtures/."""
    pdfs = list(FIXTURES_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF files in tests/fixtures/ — add at least one.")
    return pdfs[0]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPDFParser:

    def setup_method(self):
        self.parser = PDFParser()
        self.pdf_path = get_test_pdf()
        self.doc_id = make_document_id(self.pdf_path.name)

    def test_parse_returns_parsed_document(self):
        """parse() should return a ParsedDocument instance."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert isinstance(result, ParsedDocument)

    def test_parse_has_pages(self):
        """Parsed document should have at least one page."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert len(result.pages) > 0

    def test_parse_has_blocks(self):
        """Each page should have at least one text block."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert len(result.all_blocks) > 0

    def test_page_numbers_are_one_based(self):
        """Page numbers should start at 1, not 0."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        page_numbers = [page.page_number for page in result.pages]
        assert min(page_numbers) >= 1

    def test_blocks_have_bbox(self):
        """Every text block should have valid BBox coordinates."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        for block in result.all_blocks:
            assert isinstance(block, TextBlock)
            assert block.bbox["x0"] >= 0
            assert block.bbox["y0"] >= 0
            assert block.bbox["x1"] > block.bbox["x0"]
            assert block.bbox["y1"] > block.bbox["y0"]

    def test_blocks_have_page_dimensions(self):
        """Every block should have non-zero page dimensions."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        for block in result.all_blocks:
            assert block.page_width > 0
            assert block.page_height > 0

    def test_blocks_have_non_empty_text(self):
        """All extracted blocks should have non-empty text."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        for block in result.all_blocks:
            assert block.text.strip() != ""

    def test_document_id_stored(self):
        """document_id should match what was passed in."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert result.document_id == self.doc_id

    def test_filename_stored(self):
        """filename should match the actual file name."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert result.filename == self.pdf_path.name

    def test_file_size_recorded(self):
        """file_size_mb should be positive."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert result.file_size_mb > 0

    def test_full_text_property(self):
        """ParsedPage.full_text should join all block texts."""
        result = self.parser.parse(self.pdf_path, self.doc_id)
        page = result.pages[0]
        full = page.full_text
        assert isinstance(full, str)
        assert len(full) > 0

    def test_page_limit_enforced(self):
        """Parser should not exceed max_pages_per_pdf setting."""
        from core.config import settings
        result = self.parser.parse(self.pdf_path, self.doc_id)
        assert len(result.pages) <= settings.max_pages_per_pdf

    def test_nonexistent_file_raises(self):
        """Parsing a file that doesn't exist should raise PDFParseError."""
        fake_path = FIXTURES_DIR / "nonexistent.pdf"
        with pytest.raises((PDFParseError, FileNotFoundError)):
            self.parser.parse(fake_path, "fake_id")

    def test_file_too_large_raises(self):
        """Files exceeding max size should raise FileTooLargeError."""
        with patch("brain.parser.settings") as mock_settings:
            mock_settings.max_pages_per_pdf = 20
            mock_settings.max_file_size_mb = 0.000001  # Tiny to force error
            parser = PDFParser()
            with pytest.raises(FileTooLargeError):
                parser.parse(self.pdf_path, self.doc_id)