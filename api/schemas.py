"""
api/schemas.py
==============
Pydantic request/response models for all API endpoints.

Changes from previous version:
    AnswerSchema  — added rerank_score field
    QueryResponse — added reranked field
"""

from pydantic import BaseModel, Field
from datetime import datetime


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id:  str
    filename:     str
    total_pages:  int
    total_chunks: int
    total_vectors: int
    message:      str

    model_config = {"json_schema_extra": {
        "example": {
            "document_id":   "a3f1c2d4e5b6",
            "filename":      "manual_v2.pdf",
            "total_pages":   14,
            "total_chunks":  87,
            "total_vectors": 160,
            "message":       "manual_v2.pdf ingested successfully.",
        }
    }}


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:  str = Field(min_length=3, max_length=500)
    top_k:  int = Field(default=5, ge=1, le=20)

    model_config = {"json_schema_extra": {
        "example": {
            "query":  "What is the warranty period for the product?",
            "top_k":  5,
        }
    }}


class BBoxSchema(BaseModel):
    x0:          float
    y0:          float
    x1:          float
    y1:          float
    page_width:  float
    page_height: float


class AnswerSchema(BaseModel):
    text:         str   = Field(description="Extracted answer text.")
    span_score:   float = Field(description="RoBERTa span confidence score.")
    null_score:   float = Field(description="RoBERTa no-answer score.")
    rerank_score: float = Field(
        description="Cross-Encoder relevance score for the source chunk. "
                    "Higher means the chunk was judged more relevant to the query."
    )
    filename:     str   = Field(description="Source PDF filename.")
    page_number:  int   = Field(description="Source page number (1-based).")
    bbox:         BBoxSchema
    chunk_text:   str   = Field(description="Full chunk context.")
    span_hash:    str   = Field(description="SHA-256 fingerprint of normalized answer.")
    rake_used:    bool  = Field(description="True if RAKE fallback was used.")


class QueryResponse(BaseModel):
    query:                 str
    answers:               list[AnswerSchema]
    total_answers:         int
    total_chunks_searched: int
    rake_used:             bool
    reranked:              bool  = Field(
        description="True if Cross-Encoder re-ranking was applied."
    )
    processing_ms:         float

    model_config = {"json_schema_extra": {
        "example": {
            "query":                 "What is the warranty period?",
            "answers":               [],
            "total_answers":         1,
            "total_chunks_searched": 5,
            "rake_used":             False,
            "reranked":              True,
            "processing_ms":         380.4,
        }
    }}


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentSchema(BaseModel):
    document_id:  str
    filename:     str
    total_pages:  int
    total_chunks: int
    file_size_mb: float
    status:       str
    created_at:   datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentSchema]
    total:     int


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    status: str
    detail: str = ""


class HealthResponse(BaseModel):
    status:     str
    version:    str
    components: dict[str, ComponentStatus]


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:   str
    message: str
    details: dict = {}