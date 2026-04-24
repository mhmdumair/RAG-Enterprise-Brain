"""
api/middleware.py
=================
CORS, request timing, and global exception handling.

Applied to every request in the correct order:
    1. CORS         — allows Next.js frontend to call the API
    2. Timing       — adds X-Process-Time-Ms response header
    3. Error handler— converts all domain exceptions → JSON ErrorResponse
"""

import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.logger import get_logger
from core.exceptions import (
    BrainError,
    FileTooLargeError,
    TooManyPDFsError,
    PDFParseError,
    NoAnswerFoundError,
    RetrievalError,
    QAModelError,
    IngestionError,
)
from api.schemas import ErrorResponse

logger = get_logger(__name__)


def register_middleware(app: FastAPI) -> None:
    """
    Attach all middleware to the FastAPI app.
    Called once in main.py during app creation.
    """

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allows the Next.js frontend (any localhost port during dev,
    # specific domain in prod) to make cross-origin requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",   # Next.js dev server
            "http://localhost:3001",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing ─────────────────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers.
    Converts all domain exceptions into consistent JSON ErrorResponse.
    """

    # ── 413 File too large ─────────────────────────────────────────────────────
    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(request: Request, exc: FileTooLargeError):
        return JSONResponse(
            status_code=413,
            content=ErrorResponse(
                error="FileTooLargeError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    # ── 429 Too many PDFs ──────────────────────────────────────────────────────
    @app.exception_handler(TooManyPDFsError)
    async def too_many_pdfs_handler(request: Request, exc: TooManyPDFsError):
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                error="TooManyPDFsError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    # ── 422 PDF parse error ────────────────────────────────────────────────────
    @app.exception_handler(PDFParseError)
    async def pdf_parse_handler(request: Request, exc: PDFParseError):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="PDFParseError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    # ── 404 No answer found ────────────────────────────────────────────────────
    @app.exception_handler(NoAnswerFoundError)
    async def no_answer_handler(request: Request, exc: NoAnswerFoundError):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="NoAnswerFoundError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    # ── 503 Retrieval / QA model errors ───────────────────────────────────────
    @app.exception_handler(RetrievalError)
    async def retrieval_handler(request: Request, exc: RetrievalError):
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="RetrievalError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    @app.exception_handler(QAModelError)
    async def qa_model_handler(request: Request, exc: QAModelError):
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="QAModelError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    # ── 500 Generic ingestion / brain error ────────────────────────────────────
    @app.exception_handler(BrainError)
    async def brain_error_handler(request: Request, exc: BrainError):
        logger.error(
            "Unhandled BrainError",
            extra={"error": exc.message, "details": exc.details},
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=exc.__class__.__name__,
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    # ── 500 Catch-all ─────────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            extra={"error": str(exc), "type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred.",
                details={"type": type(exc).__name__},
            ).model_dump(),
        )