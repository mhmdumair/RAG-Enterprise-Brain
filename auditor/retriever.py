"""
auditor/retriever.py
====================
ANN retrieval — embed query → FAISS search → MongoDB chunk lookup.

This is the first stage of the auditor pipeline.
It bridges Boundary 1 (FAISS + MongoDB) and Boundary 2 (QA).

Flow:
    query string
      └─ Embedder.embed_query()          → query vector (384,)
      └─ FAISSIndex.search()             → top-K vector_ids + distances
      └─ get_chunks_by_vector_ids()      → chunk metadata from MongoDB
      └─ list[RetrievedChunk]            → passed to worker/dispatcher
"""

from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import RetrievalError
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from db.queries import get_chunks_by_vector_ids

logger = get_logger(__name__)


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A chunk returned by ANN search, enriched with MongoDB metadata.

    Fields:
        vector_id    — FAISS index position
        distance     — L2 distance from query vector (lower = more similar)
        chunk_id     — MongoDB chunk _id
        document_id  — parent document ID
        filename     — source PDF filename
        page_number  — source page (1-based)
        text         — chunk text (used as QA context)
        bbox         — normalized BBox for frontend highlighting
    """
    vector_id: int
    distance: float
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    bbox: dict


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Retrieves the top-K most relevant chunks for a query.

    Usage:
        retriever = Retriever(embedder, faiss_index, db)
        chunks = await retriever.retrieve("What is the warranty period?")
    """

    def __init__(
        self,
        embedder: Embedder,
        faiss_index: FAISSIndex,
        db: AsyncIOMotorDatabase,
    ):
        self._embedder = embedder
        self._faiss = faiss_index
        self._db = db

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Embed query and return top-K enriched chunks.

        Args:
            query — the audit question string
            k     — override top-K (defaults to settings.top_k_chunks)

        Returns:
            list[RetrievedChunk] ordered by relevance (closest first)

        Raises:
            RetrievalError — if embedding or FAISS search fails
        """
        k = k or settings.top_k_chunks

        if not self._faiss.is_ready:
            raise RetrievalError("FAISS index is empty. Ingest documents first.")

        # ── Step 1: Embed query ───────────────────────────────────────────────
        try:
            query_vector = self._embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(f"Query embedding failed: {exc}") from exc

        # ── Step 2: ANN search ────────────────────────────────────────────────
        try:
            vector_ids, distances = self._faiss.search(query_vector, k=k)
        except Exception as exc:
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

        if not vector_ids:
            logger.warning("FAISS returned no results", extra={"query": query})
            return []

        # ── Step 3: MongoDB chunk lookup ──────────────────────────────────────
        try:
            raw_chunks = await get_chunks_by_vector_ids(self._db, vector_ids)
        except Exception as exc:
            raise RetrievalError(f"MongoDB lookup failed: {exc}") from exc

        # Build a distance map for ordering (FAISS order may differ from Mongo)
        distance_map = dict(zip(vector_ids, distances))

        # ── Step 4: Build RetrievedChunk objects ──────────────────────────────
        retrieved: list[RetrievedChunk] = []
        for raw in raw_chunks:
            vid = raw.get("vector_id")
            retrieved.append(RetrievedChunk(
                vector_id=vid,
                distance=distance_map.get(vid, 9999.0),
                chunk_id=raw.get("chunk_id", ""),
                document_id=raw.get("document_id", ""),
                filename=raw.get("filename", ""),
                page_number=raw.get("page_number", 0),
                text=raw.get("text", ""),
                bbox=raw.get("bbox", {}),
            ))

        # Sort by distance ascending (most relevant first)
        retrieved.sort(key=lambda c: c.distance)

        logger.info(
            "Retrieval complete",
            extra={
                "query": query[:60],
                "k": k,
                "retrieved": len(retrieved),
            },
        )

        return retrieved