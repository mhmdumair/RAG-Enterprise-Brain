"""
api/main.py
===========
FastAPI application entry point.

Responsibilities:
  1. Create the FastAPI app instance
  2. Register middleware (CORS, timing)
  3. Register exception handlers
  4. Lifespan startup — load models, FAISS index, create singletons
  5. Lifespan shutdown — close DB, shut down thread pool
  6. Mount all route routers

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.config import settings
from core.logger import get_logger
from db.client import get_database, ping_database, close_client
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from brain.pipeline import IngestionPipeline
from auditor.dispatcher import AuditDispatcher
from api.middleware import register_middleware, register_exception_handlers
from api.dependencies import (
    set_embedder,
    set_faiss_index,
    set_dispatcher,
    set_pipeline,
)
from api.routes import health, documents, ingest, query

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Everything before 'yield' runs at startup.
    Everything after 'yield' runs at shutdown.

    Startup order matters:
        1. Verify MongoDB connection
        2. Load embedding model (slow — ~5s first time)
        3. Load or initialize FAISS index
        4. Load QA model via dispatcher (slow — ~3s first time)
        5. Create pipeline instance
    """
    logger.info("Enterprise Brain starting up...")

    # ── 1. MongoDB ────────────────────────────────────────────────────────────
    db = get_database()
    mongo_ok = await ping_database()
    if not mongo_ok:
        logger.error("MongoDB unavailable at startup — continuing anyway.")
    else:
        logger.info("MongoDB connected.")

    # ── 2. Embedding model ────────────────────────────────────────────────────
    logger.info("Loading embedding model...")
    embedder = Embedder()
    set_embedder(embedder)
    logger.info("Embedding model ready.")

    # ── 3. FAISS index ────────────────────────────────────────────────────────
    logger.info("Loading FAISS index...")
    faiss_index = FAISSIndex()
    loaded = faiss_index.load()
    if loaded:
        logger.info(
            "FAISS index loaded from disk.",
            extra={"vectors": faiss_index.total_vectors},
        )
    else:
        logger.warning(
            "No FAISS index on disk — index will be built after first ingest."
        )
    set_faiss_index(faiss_index)

    # ── 4. QA dispatcher (loads QA model internally) ──────────────────────────
    logger.info("Loading QA model via dispatcher...")
    dispatcher = AuditDispatcher(embedder, faiss_index, db)
    set_dispatcher(dispatcher)
    logger.info("QA model ready.")

    # ── 5. Ingestion pipeline ─────────────────────────────────────────────────
    pipeline = IngestionPipeline(db, faiss_index)
    set_pipeline(pipeline)
    logger.info("Ingestion pipeline ready.")

    logger.info(
        "Enterprise Brain startup complete.",
        extra={
            "host": settings.api_host,
            "port": settings.api_port,
            "version": settings.api_version,
        },
    )

    yield  # ── App is running ─────────────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Enterprise Brain shutting down...")
    dispatcher.shutdown()
    await close_client()
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Zero-LLM Multi-Agent Deterministic RAG system. "
            "Extracts verified answer spans from ingested PDFs "
            "with full source attribution and hallucination-free guarantees."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    register_middleware(app)
    register_exception_handlers(app)

    # ── Mount routers ─────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    return app


app = create_app()