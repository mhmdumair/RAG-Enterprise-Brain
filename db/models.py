"""
db/models.py
============
Data models for each MongoDB collection.
These are plain dataclasses — not ORM models.
They define the exact shape of every document stored in MongoDB.

Collections:
    documents  — one record per ingested PDF
    chunks     — one record per text chunk / FAISS vector
    results    — one record per completed audit query

Changes from previous version:
    ChunkRecord — added is_fragment and sentence_count fields.
    All other fields and method signatures are identical.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from core.utils import NormalizedBBox


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Documents collection ──────────────────────────────────────────────────────

@dataclass
class DocumentRecord:
    """
    One record per ingested PDF.
    Stored in the 'documents' collection.

    Fields:
        document_id   — deterministic SHA-256 hash of filename (12 chars)
        filename      — original uploaded filename
        total_pages   — number of pages processed
        total_chunks  — total text chunks extracted
        file_size_mb  — file size at upload time
        status        — 'ingested' | 'failed'
        created_at    — UTC timestamp of ingestion
    """
    document_id:  str
    filename:     str
    total_pages:  int
    total_chunks: int
    file_size_mb: float
    status:       str      = "ingested"
    created_at:   datetime = field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        d = asdict(self)
        d["_id"] = self.document_id
        return d


# ── Chunks collection ─────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """
    One record per text chunk extracted from a PDF page.
    Stored in the 'chunks' collection.

    The vector_id is the integer index position in the FAISS index.
    This is what ties the FAISS search result back to the source document.

    Fields:
        chunk_id       — deterministic ID: sha256(doc_id:page:idx)[:16]
        vector_id      — integer position in the FAISS index
        document_id    — parent document reference
        filename       — original PDF filename (denormalized for fast lookup)
        page_number    — 1-based page number
        chunk_index    — position within the document (0-based)
        text           — the raw chunk text (used as QA context)
        bbox           — normalized bounding box (0.0–1.0 coordinates)
                         union of all source blocks in this chunk
        is_fragment    — True if chunk was produced by force-splitting
                         an oversized sentence (rare)
        sentence_count — number of complete sentences in this chunk
        created_at     — UTC timestamp
    """
    chunk_id:       str
    vector_id:      int
    document_id:    str
    filename:       str
    page_number:    int
    chunk_index:    int
    text:           str
    bbox:           NormalizedBBox
    is_fragment:    bool     = False
    sentence_count: int      = 0
    created_at:     datetime = field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        d = asdict(self)
        d["_id"]  = self.chunk_id
        d["bbox"] = dict(self.bbox)
        return d


# ── Results collection ────────────────────────────────────────────────────────

@dataclass
class AnswerSpan:
    """
    A single verified answer span within a result.

    Fields:
        text       — the extracted answer string
        score      — confidence score from RoBERTa (0.0–1.0)
        filename   — source PDF filename
        page_number— source page (1-based)
        bbox       — normalized BBox for frontend highlighting
        chunk_text — full chunk context the answer was extracted from
        span_hash  — SHA-256 of normalized text (for deduplication)
    """
    text:        str
    score:       float
    filename:    str
    page_number: int
    bbox:        NormalizedBBox
    chunk_text:  str
    span_hash:   str


@dataclass
class ResultRecord:
    """
    One record per audit query.
    Stored in the 'results' collection.

    Fields:
        query_hash            — SHA-256 of the query string
        query                 — original query string
        spans                 — list of verified AnswerSpan objects
        rake_used             — True if RAKE fallback was triggered
        total_chunks_searched — how many chunks were evaluated
        created_at            — UTC timestamp
    """
    query_hash:             str
    query:                  str
    spans:                  list[AnswerSpan]
    rake_used:              bool
    total_chunks_searched:  int
    created_at:             datetime = field(default_factory=_now)

    def to_mongo(self) -> dict[str, Any]:
        d = asdict(self)
        d["_id"] = f"{self.query_hash}_{self.created_at.timestamp()}"
        return d