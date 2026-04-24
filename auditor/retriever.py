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

from dataclasses import dataclass, field
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import RetrievalError
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from db.queries import get_chunks_by_vector_ids, get_chunks_by_document_and_range

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
        chunk_index  — position in document (for context stitching)
        stitched_text — context-stitched version (if enabled)
    """
    vector_id: int
    distance: float
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    bbox: dict
    chunk_index: int = 0
    stitched_text: str = field(default="")
    stitched_chunks_count: int = 1


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
        stitch_context: bool = True,
        window_size: int = 1,
    ) -> list[RetrievedChunk]:
        """
        Embed query and return top-K enriched chunks.

        Args:
            query — the audit question string
            k     — override top-K (defaults to settings.top_k_chunks)
            stitch_context — if True, fetch neighboring chunks for context
            window_size — number of neighbor chunks to fetch on each side

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
                chunk_index=raw.get("chunk_index", 0),
            ))

        # Sort by distance ascending (most relevant first)
        retrieved.sort(key=lambda c: c.distance)

        # ── Step 5: Context Stitching (The key improvement!) ──────────────────
        if stitch_context:
            retrieved = await self._stitch_context(retrieved, window_size)

        logger.info(
            "Retrieval complete",
            extra={
                "query": query[:60],
                "k": k,
                "retrieved": len(retrieved),
                "stitched": stitch_context,
            },
        )

        return retrieved

    async def _stitch_context(
        self, 
        chunks: list[RetrievedChunk], 
        window_size: int = 1
    ) -> list[RetrievedChunk]:
        """
        Stitch neighboring chunks to provide broader context.
        
        For each retrieved chunk, fetches window_size chunks before and after
        to create a super-chunk that bridges the context gap.
        This allows the QA model to see both the question setup and answer.
        """
        if not chunks:
            return chunks

        stitched_chunks = []
        
        for chunk in chunks:
            try:
                # Fetch neighboring chunks from the same document
                neighbors = await get_chunks_by_document_and_range(
                    self._db,
                    document_id=chunk.document_id,
                    start_index=max(0, chunk.chunk_index - window_size),
                    end_index=chunk.chunk_index + window_size + 1,
                )
                
                if neighbors and len(neighbors) > 1:
                    # Sort by chunk_index to maintain document order
                    neighbors.sort(key=lambda x: x.get("chunk_index", 0))
                    
                    # Stitch texts together in order
                    stitched_text = " ".join([n.get("text", "") for n in neighbors])
                    
                    # Store stitched version
                    chunk.stitched_text = stitched_text
                    chunk.stitched_chunks_count = len(neighbors)
                    
                    logger.debug(
                        "Stitched context",
                        extra={
                            "chunk_id": chunk.chunk_id,
                            "original_index": chunk.chunk_index,
                            "neighbors": len(neighbors),
                            "original_length": len(chunk.text),
                            "stitched_length": len(stitched_text),
                        }
                    )
                else:
                    chunk.stitched_text = chunk.text
                    chunk.stitched_chunks_count = 1
                    
            except Exception as exc:
                logger.warning(
                    f"Failed to stitch context for chunk {chunk.chunk_id}: {exc}"
                )
                chunk.stitched_text = chunk.text
                chunk.stitched_chunks_count = 1
            
            stitched_chunks.append(chunk)
        
        return stitched_chunks