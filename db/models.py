"""
db/models.py
============
Data models for each MongoDB collection.

Changes from previous version:
    ChunkRecord — added next_chunk_id, prev_chunk_id, is_linked fields.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from core.utils import NormalizedBBox


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Documents collection ──────────────────────────────────────────────────────

@dataclass
class DocumentRecord:
    """One record per ingested PDF."""
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    file_size_mb: float
    status: str = "ingested"
    created_at: datetime = field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        d = asdict(self)
        d["_id"] = self.document_id
        return d


# ── Chunks collection ─────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """
    One record per text chunk extracted from a PDF page.

    Fields:
        chunk_id       — deterministic ID
        vector_id      — integer position in the FAISS index
        document_id    — parent document reference
        filename       — original PDF filename
        page_number    — 1-based page number
        chunk_index    — position within the document (0-based)
        text           — the raw chunk text
        bbox           — normalized bounding box
        is_fragment    — True if chunk was produced by force-splitting
        sentence_count — number of complete sentences in this chunk
        next_chunk_id  — ID of next chunk in sequence (if sentence spans)
        prev_chunk_id  — ID of previous chunk in sequence (if sentence spans)
        is_linked      — True if this chunk is part of a link chain
        created_at     — UTC timestamp
    """
    chunk_id: str
    vector_id: int
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    bbox: NormalizedBBox
    is_fragment: bool = False
    sentence_count: int = 0
    next_chunk_id: Optional[str] = None
    prev_chunk_id: Optional[str] = None
    is_linked: bool = False
    created_at: datetime = field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        d = asdict(self)
        d["_id"] = self.chunk_id
        d["bbox"] = dict(self.bbox)
        return d


# ── Results collection ────────────────────────────────────────────────────────

@dataclass
class AnswerSpan:
    """A single verified answer span within a result."""
    text: str
    score: float
    filename: str
    page_number: int
    bbox: NormalizedBBox
    chunk_text: str
    span_hash: str


@dataclass
class ResultRecord:
    """One record per audit query."""
    query_hash: str
    query: str
    spans: list[AnswerSpan]
    rake_used: bool
    total_chunks_searched: int
    created_at: datetime = field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        d = asdict(self)
        d["_id"] = f"{self.query_hash}_{self.created_at.timestamp()}"
        return d