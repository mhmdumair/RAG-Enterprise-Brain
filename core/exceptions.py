"""
core/exceptions.py
==================
All custom domain exceptions for the Enterprise Brain.

Hierarchy:
    BrainError                  (base)
    ├── IngestionError          (Boundary 1 — PDF parsing / indexing)
    │   ├── FileTooLargeError
    │   ├── TooManyPDFsError
    │   ├── PDFParseError
    │   └── EmbeddingError
    ├── AuditorError            (Boundary 2 — QA / retrieval)
    │   ├── RetrievalError
    │   ├── QAModelError
    │   └── NoAnswerFoundError
    └── StorageError            (DB / FAISS persistence)
        ├── DatabaseError
        └── IndexError
"""


class BrainError(Exception):
    """Base exception for all Enterprise Brain errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details})"


# ── Ingestion errors (Boundary 1) ─────────────────────────────────────────────

class IngestionError(BrainError):
    """Raised when the ingestion pipeline fails."""


class FileTooLargeError(IngestionError):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, filename: str, size_mb: float, limit_mb: int):
        super().__init__(
            message=f"File '{filename}' is {size_mb:.1f}MB — exceeds the {limit_mb}MB limit.",
            details={"filename": filename, "size_mb": size_mb, "limit_mb": limit_mb},
        )


class TooManyPDFsError(IngestionError):
    """Raised when ingestion would exceed the max PDF count."""

    def __init__(self, current: int, limit: int):
        super().__init__(
            message=f"Cannot ingest: {current} PDFs already ingested, limit is {limit}.",
            details={"current": current, "limit": limit},
        )


class PDFParseError(IngestionError):
    """Raised when PyMuPDF cannot parse a file."""

    def __init__(self, filename: str, reason: str):
        super().__init__(
            message=f"Failed to parse '{filename}': {reason}",
            details={"filename": filename, "reason": reason},
        )


class EmbeddingError(IngestionError):
    """Raised when the embedding model fails to encode a chunk."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Embedding failed: {reason}",
            details={"reason": reason},
        )


# ── Auditor errors (Boundary 2) ───────────────────────────────────────────────

class AuditorError(BrainError):
    """Raised when the extractive QA auditor fails."""


class RetrievalError(AuditorError):
    """Raised when FAISS ANN search fails."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Retrieval failed: {reason}",
            details={"reason": reason},
        )


class QAModelError(AuditorError):
    """Raised when the QA model fails during inference."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"QA model error: {reason}",
            details={"reason": reason},
        )


class NoAnswerFoundError(AuditorError):
    """
    Raised when no chunk produces a span that passes the
    abstention threshold (S_span > S_null + tau_ans),
    even after the RAKE fallback retry.
    """

    def __init__(self, query: str):
        super().__init__(
            message=f"No verified answer found for query: '{query}'",
            details={"query": query},
        )


# ── Storage errors ────────────────────────────────────────────────────────────

class StorageError(BrainError):
    """Raised when persistence operations fail."""


class DatabaseError(StorageError):
    """Raised when a MongoDB operation fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Database error during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )


class IndexError(StorageError):
    """Raised when FAISS index operations fail."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"FAISS index error during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )