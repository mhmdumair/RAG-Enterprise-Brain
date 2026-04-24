"""
api/routes/health.py
====================
GET /health — system health check endpoint.

Checks:
  - MongoDB: can we ping the database?
  - FAISS index: is it loaded and non-empty?
  - Embedder: is the model initialized?
  - Dispatcher (QA model): is it initialized?

Returns 200 if all components are healthy.
Returns 503 if any critical component is down.
"""

from fastapi import APIRouter, Depends

from core.config import settings
from core.logger import get_logger
from db.client import ping_database
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from auditor.dispatcher import AuditDispatcher
from api.schemas import HealthResponse, ComponentStatus
from api.dependencies import get_embedder, get_faiss_index, get_dispatcher

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"],
)
async def health_check(
    embedder: Embedder = Depends(get_embedder),
    faiss_index: FAISSIndex = Depends(get_faiss_index),
    dispatcher: AuditDispatcher = Depends(get_dispatcher),
):
    """
    Check all system components and return their status.

    Returns 200 if status is 'ok'.
    Returns 503 if status is 'degraded'.
    """
    components: dict[str, ComponentStatus] = {}
    all_ok = True

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongo_ok = await ping_database()
    components["mongodb"] = ComponentStatus(
        status="ok" if mongo_ok else "error",
        detail="" if mongo_ok else "MongoDB ping failed.",
    )
    if not mongo_ok:
        all_ok = False

    # ── FAISS index ───────────────────────────────────────────────────────────
    faiss_ok = faiss_index.is_ready
    components["faiss_index"] = ComponentStatus(
        status="ok" if faiss_ok else "error",
        detail=f"{faiss_index.total_vectors} vectors" if faiss_ok else "Index empty or not loaded.",
    )
    if not faiss_ok:
        all_ok = False

    # ── Embedder ──────────────────────────────────────────────────────────────
    components["embedder"] = ComponentStatus(
        status="ok" if embedder is not None else "error",
        detail=settings.embedding_model_name,
    )

    # ── QA model (via dispatcher) ─────────────────────────────────────────────
    components["qa_model"] = ComponentStatus(
        status="ok" if dispatcher is not None else "error",
        detail=settings.qa_model_name,
    )

    overall = "ok" if all_ok else "degraded"

    logger.info("Health check", extra={"status": overall})

    from fastapi.responses import JSONResponse
    response_data = HealthResponse(
        status=overall,
        version=settings.api_version,
        components=components,
    )

    if not all_ok:
        return JSONResponse(status_code=503, content=response_data.model_dump())

    return response_data