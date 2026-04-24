"""
api/schemas.py
==============
Pydantic request/response models for all API endpoints.

Every endpoint uses these models — nothing returns raw dicts.
This gives automatic validation, serialization, and OpenAPI docs.

Endpoints:
    POST /ingest     → IngestResponse
    POST /query      → QueryRequest, QueryResponse
    GET  /documents  → DocumentListResponse
    GET  /health     → HealthResponse
"""

from pydantic import BaseModel, Field
from datetime import datetime


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Returned after a successful PDF ingestion."""

    document_id: str = Field(description="Deterministic SHA-256 ID of the document.")
    filename: str = Field(description="Original uploaded filename.")
    total_pages: int = Field(description="Number of pages processed.")
    total_chunks: int = Field(description="Number of text chunks created.")
    total_vectors: int = Field(description="Total vectors now in the FAISS index.")
    message: str = Field(description="Human-readable status message.")

    model_config = {"json_schema_extra": {
        "example": {
            "document_id": "a3f1c2d4e5b6",
            "filename": "manual_v2.pdf",
            "total_pages": 14,
            "total_chunks": 87,
            "total_vectors": 160,
            "message": "manual_v2.pdf ingested successfully.",
        }
    }}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(
        min_length=3,
        max_length=500,
        description="The audit question to answer from ingested documents.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for QA (overrides config default).",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is the warranty period for the product?",
            "top_k": 5,
        }
    }}


class BBoxSchema(BaseModel):
    """Normalized bounding box coordinates (0.0–1.0)."""
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float


class AnswerSchema(BaseModel):
    """A single verified answer span with full source attribution."""

    text: str = Field(description="The extracted answer text.")
    span_score: float = Field(description="RoBERTa confidence score for this span.")
    null_score: float = Field(description="RoBERTa no-answer score (for reference).")
    filename: str = Field(description="Source PDF filename.")
    page_number: int = Field(description="Source page number (1-based).")
    bbox: BBoxSchema = Field(description="Normalized BBox for frontend highlighting.")
    chunk_text: str = Field(description="Full chunk context the answer was extracted from.")
    span_hash: str = Field(description="SHA-256 fingerprint of the normalized answer.")
    rake_used: bool = Field(description="True if RAKE fallback was used for this answer.")


class QueryResponse(BaseModel):
    """Returned after a successful audit query."""

    query: str = Field(description="The original query string.")
    answers: list[AnswerSchema] = Field(description="Ranked list of verified answer spans.")
    total_answers: int = Field(description="Number of unique verified answers found.")
    total_chunks_searched: int = Field(description="Total chunks evaluated during QA.")
    rake_used: bool = Field(description="True if RAKE fallback was triggered.")
    processing_ms: float = Field(description="Total processing time in milliseconds.")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is the warranty period?",
            "answers": [{
                "text": "two years",
                "span_score": 3.51,
                "null_score": 0.09,
                "filename": "manual_v2.pdf",
                "page_number": 14,
                "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.25,
                         "page_width": 595.0, "page_height": 842.0},
                "chunk_text": "The product warranty period is two years from purchase.",
                "span_hash": "a3f1c2...",
                "rake_used": False,
            }],
            "total_answers": 1,
            "total_chunks_searched": 5,
            "rake_used": False,
            "processing_ms": 412.3,
        }
    }}


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentSchema(BaseModel):
    """Summary of one ingested document."""

    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    file_size_mb: float
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Returned by GET /documents."""

    documents: list[DocumentSchema]
    total: int = Field(description="Total number of ingested documents.")


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    """Status of a single system component."""
    status: str = Field(description="'ok' or 'error'")
    detail: str = Field(default="", description="Extra info if status is 'error'.")


class HealthResponse(BaseModel):
    """Returned by GET /health."""

    status: str = Field(description="'ok' if all components healthy, else 'degraded'.")
    version: str = Field(description="API version string.")
    components: dict[str, ComponentStatus] = Field(
        description="Per-component health status."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "status": "ok",
            "version": "0.1.0",
            "components": {
                "mongodb": {"status": "ok", "detail": ""},
                "faiss_index": {"status": "ok", "detail": "160 vectors"},
                "qa_model": {"status": "ok", "detail": ""},
                "embedder": {"status": "ok", "detail": ""},
            }
        }
    }}


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned on all 4xx/5xx responses."""

    error: str = Field(description="Error type name.")
    message: str = Field(description="Human-readable error description.")
    details: dict = Field(default={}, description="Additional structured error info.")

    model_config = {"json_schema_extra": {
        "example": {
            "error": "NoAnswerFoundError",
            "message": "No verified answer found for query: 'What is X?'",
            "details": {"query": "What is X?"},
        }
    }}