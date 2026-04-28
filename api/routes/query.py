"""
api/routes/query.py
===================
POST /query — run an audit question against ingested documents.

Change from previous version:
    QueryResponse now includes reranked field from DispatchResult.
"""

import time
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

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
    request:    QueryRequest,
    dispatcher: AuditDispatcher = Depends(get_dispatcher),
):
    """
    Run an audit question against all ingested documents.

    Returns ranked verified answer spans, each attributed to its
    source document, page number, and exact bounding box coordinates.

    The reranked field indicates whether Cross-Encoder re-ranking
    was applied to improve answer quality.

    Returns 404 if no answer passes the confidence threshold.
    """
    start_time = time.perf_counter()

    logger.info(
        "Query received",
        extra={
            "query":  request.query[:80],
            "top_k":  request.top_k,
        },
    )

    # ── Run dispatch ──────────────────────────────────────────────────────────
    try:
        result = await dispatcher.dispatch(request.query)
    except NoAnswerFoundError as exc:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="NoAnswerFoundError",
                message=exc.message,
                details=exc.details,
            ).model_dump(),
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # ── Build response ────────────────────────────────────────────────────────
    answers = [
        AnswerSchema(
            text=ans.text,
            span_score=ans.span_score,
            null_score=ans.null_score,
            rerank_score=ans.rerank_score,
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
            "query":    request.query[:80],
            "answers":  len(answers),
            "reranked": result.reranked,
            "ms":       round(elapsed_ms, 1),
        },
    )

    return QueryResponse(
        query=result.query,
        answers=answers,
        total_answers=len(answers),
        total_chunks_searched=result.total_chunks_searched,
        rake_used=result.rake_used,
        reranked=result.reranked,
        processing_ms=round(elapsed_ms, 2),
    )