"""
core/utils.py
=============
Pure utility functions — no business logic, no side effects.
Safe to import from anywhere.
"""

import hashlib
import re
import unicodedata
from typing import TypedDict


# ── BBox types ────────────────────────────────────────────────────────────────

class RawBBox(TypedDict):
    """Pixel-space bounding box as returned by PyMuPDF."""
    x0: float
    y0: float
    x1: float
    y1: float


class NormalizedBBox(TypedDict):
    """
    Percentage-space bounding box (0.0 – 1.0).
    Used by the frontend to draw SVG highlight rectangles
    regardless of the rendered PDF zoom level.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float


# ── BBox helpers ──────────────────────────────────────────────────────────────

def normalize_bbox(
    bbox: RawBBox,
    page_width: float,
    page_height: float,
) -> NormalizedBBox:
    """
    Convert pixel-space BBox coordinates to 0.0–1.0 percentages.

    Args:
        bbox:         Raw BBox from PyMuPDF  {x0, y0, x1, y1}
        page_width:   Page width in points (from page.rect.width)
        page_height:  Page height in points (from page.rect.height)

    Returns:
        NormalizedBBox with all coords as fractions of page dimensions.

    Example:
        >>> normalize_bbox({"x0": 50, "y0": 100, "x1": 200, "y1": 150}, 500, 700)
        {"x0": 0.1, "y0": 0.143, "x1": 0.4, "y1": 0.214, ...}
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Invalid page dimensions: {page_width}x{page_height}")

    return NormalizedBBox(
        x0=round(bbox["x0"] / page_width, 6),
        y0=round(bbox["y0"] / page_height, 6),
        x1=round(bbox["x1"] / page_width, 6),
        y1=round(bbox["y1"] / page_height, 6),
        page_width=page_width,
        page_height=page_height,
    )


def bbox_area(bbox: RawBBox) -> float:
    """Return the pixel area of a bounding box."""
    return max(0.0, bbox["x1"] - bbox["x0"]) * max(0.0, bbox["y1"] - bbox["y0"])


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize raw PDF text for embedding and QA.

    Steps:
      1. Unicode NFKC normalization (e.g. ligatures → ascii)
      2. Replace non-breaking spaces and tabs with regular spaces
      3. Collapse multiple whitespace into single space
      4. Strip leading/trailing whitespace

    Args:
        text: Raw string extracted from PyMuPDF

    Returns:
        Cleaned string ready for the embedder and QA model.
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Replace special whitespace
    text = text.replace("\xa0", " ").replace("\t", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to max_chars, cutting at the last word boundary.
    Avoids splitting mid-word.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


# ── Hashing ───────────────────────────────────────────────────────────────────

def sha256_hash(text: str) -> str:
    """
    Return the SHA-256 hex digest of a string.
    Used by the deduplicator to fingerprint answer spans.

    Args:
        text: The string to hash (normalized answer span)

    Returns:
        64-character lowercase hex string.

    Example:
        >>> sha256_hash("two years")
        "a3f1..."
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_span(text: str) -> str:
    """
    Normalize an answer span before hashing for deduplication.
    Lowercases and strips punctuation so "Two years." and "two years"
    hash to the same value.
    """
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


# ── ID generation ─────────────────────────────────────────────────────────────

def make_chunk_id(document_id: str, page_number: int, chunk_index: int) -> str:
    """
    Generate a deterministic chunk ID from its coordinates.
    Used as the vector_id in FAISS and the _id in MongoDB chunks collection.

    Format: sha256(document_id:page:chunk_index)[:16]
    """
    raw = f"{document_id}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_document_id(filename: str) -> str:
    """
    Generate a deterministic document ID from the filename.
    Format: sha256(filename)[:12]
    """
    return hashlib.sha256(filename.encode()).hexdigest()[:12]