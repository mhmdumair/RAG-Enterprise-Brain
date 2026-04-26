"""
api/routes/query.py
===================
POST /query — run an audit question against ingested documents.

Flow:
    1. Validate QueryRequest (query string, top_k)
    2. Run AuditDispatcher.dispatch()
    3. Build QueryResponse with verified VerifiedAnswer spans
    4. Return with processing time

Returns 404 if no verified answer found (NoAnswerFoundError).
"""

import time
from fastapi import APIRouter, Depends

from core.logger import get_logger
from core.exceptions import NoAnswerFoundError
from auditor.dispatcher import AuditDispatcher
from api.schemas import (
    QueryRequest,
    QueryResponse,
    AnswerSchema,
    BBoxSchema,
    ErrorResponse,
)
from api.dependencies import get_dispatcher
from fastapi.responses import JSONResponse

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Run an audit query",
    tags=["Query"],
    responses={
        404: {"model": ErrorResponse, "description": "No verified answer found."},
        503: {"model": ErrorResponse, "description": "Model or retrieval error."},
    },
)
async def run_query(
    request: QueryRequest,
    dispatcher: AuditDispatcher = Depends(get_dispatcher),
):
    """
    Run an audit question against all ingested documents.

    Returns ranked verified answer spans, each attributed to its
    source document, page number, and exact bounding box coordinates.

    If no answer passes the confidence threshold (τ_ans), returns 404.
    """
    start_time = time.perf_counter()

    logger.info(
        "Query received",
        extra={"query": request.query[:80], "top_k": request.top_k},
    )

    # ── Run dispatch ──────────────────────────────────────────────────────────
    try:
        result = await dispatcher.dispatch(request.query)
    except NoAnswerFoundError as exc:
        # Return 200 with empty answers instead of 404
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            query=request.query,
            answers=[],
            total_answers=0,
            total_chunks_searched=0,
            rake_used=False,
            processing_ms=round(elapsed_ms, 2),
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # ── Build response ────────────────────────────────────────────────────────
    answers = [
        AnswerSchema(
            text=ans.text,
            span_score=ans.span_score,
            null_score=ans.null_score,
            filename=ans.filename,
            page_number=ans.page_number,
            bbox=BBoxSchema(**ans.bbox),
            chunk_text=ans.chunk_text,
            span_hash=ans.span_hash,
            rake_used=ans.rake_used,
        )
        for ans in result.answers
    ]

    logger.info(
        "Query complete",
        extra={
            "query": request.query[:80],
            "answers": len(answers),
            "ms": round(elapsed_ms, 1),
        },
    )

    return QueryResponse(
        query=result.query,
        answers=answers,
        total_answers=len(answers),
        total_chunks_searched=result.total_chunks_searched,
        rake_used=result.rake_used,
        processing_ms=round(elapsed_ms, 2),
    )