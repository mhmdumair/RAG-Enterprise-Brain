"""
api/dependencies.py
===================
FastAPI dependency injection — shared singleton instances.

All expensive objects (models, FAISS index, DB connection) are
created once at app startup and injected into routes via Depends().

This means:
  - Models load once — not per request
  - FAISS index stays in memory — not reloaded per request
  - DB connection pool is shared — not reopened per request

Usage in routes:
    @router.post("/query")
    async def query(
        request: QueryRequest,
        dispatcher: AuditDispatcher = Depends(get_dispatcher),
        db: AsyncIOMotorDatabase = Depends(get_db),
    ):
        ...
"""

from functools import lru_cache
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.logger import get_logger
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from brain.pipeline import IngestionPipeline
from auditor.dispatcher import AuditDispatcher
from db.client import get_database as _get_database

logger = get_logger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
# These are set during the FastAPI lifespan startup event in main.py
# and accessed by the Depends() functions below.

_embedder: Embedder | None = None
_faiss_index: FAISSIndex | None = None
_dispatcher: AuditDispatcher | None = None
_pipeline: IngestionPipeline | None = None


# ── Setters (called from lifespan in main.py) ─────────────────────────────────

def set_embedder(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder


def set_faiss_index(index: FAISSIndex) -> None:
    global _faiss_index
    _faiss_index = index


def set_dispatcher(dispatcher: AuditDispatcher) -> None:
    global _dispatcher
    _dispatcher = dispatcher


def set_pipeline(pipeline: IngestionPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


# ── Getters (used as FastAPI Depends) ─────────────────────────────────────────

def get_db() -> AsyncIOMotorDatabase:
    """Return the shared Motor database handle."""
    return _get_database()


def get_embedder() -> Embedder:
    if _embedder is None:
        raise RuntimeError("Embedder not initialized. Check startup lifespan.")
    return _embedder


def get_faiss_index() -> FAISSIndex:
    if _faiss_index is None:
        raise RuntimeError("FAISS index not initialized. Check startup lifespan.")
    return _faiss_index


def get_dispatcher() -> AuditDispatcher:
    if _dispatcher is None:
        raise RuntimeError("Dispatcher not initialized. Check startup lifespan.")
    return _dispatcher


def get_pipeline() -> IngestionPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Check startup lifespan.")
    return _pipeline