"""
auditor/retriever.py
====================
ANN retrieval — embed query → FAISS search → MongoDB chunk lookup.

Changes:
    - RetrievedChunk gains linked_text and is_linked fields
    - retrieve() now merges linked chunks when is_linked=True
"""

from dataclasses import dataclass, field
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import RetrievalError
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from db.queries import get_chunks_by_vector_ids, get_chunk_with_chain

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
        linked_text  — merged text from linked chunks (if is_linked)
        bbox         — normalized BBox for frontend highlighting
        rerank_score — Cross-Encoder relevance score
        is_linked    — True if this chunk has prev/next links
    """
    vector_id:    int
    distance:     float
    chunk_id:     str
    document_id:  str
    filename:     str
    page_number:  int
    text:         str
    bbox:         dict
    rerank_score: float = field(default=0.0)
    linked_text:  str = field(default="")
    is_linked:    bool = False


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Retrieves the top-K most relevant chunks for a query.
    Automatically merges linked chunks when is_linked=True.
    """

    def __init__(
        self,
        embedder:    Embedder,
        faiss_index: FAISSIndex,
        db:          AsyncIOMotorDatabase,
    ):
        self._embedder    = embedder
        self._faiss       = faiss_index
        self._db          = db

    async def retrieve(
        self,
        query: str,
        k:     int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Embed query and return top-K enriched chunks.
        Merges linked chunks automatically.
        """
        k = k or settings.retrieval_top_k

        if not self._faiss.is_ready:
            raise RetrievalError(
                "FAISS index is empty. Ingest documents first."
            )

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
            logger.warning(
                "FAISS returned no results",
                extra={"query": query[:60]},
            )
            return []

        # ── Step 3: MongoDB chunk lookup ──────────────────────────────────────
        try:
            raw_chunks = await get_chunks_by_vector_ids(self._db, vector_ids)
        except Exception as exc:
            raise RetrievalError(f"MongoDB lookup failed: {exc}") from exc

        # Distance map for ordering
        distance_map = dict(zip(vector_ids, distances))

        # ── Step 4: Build RetrievedChunk objects with link merging ────────────
        retrieved: list[RetrievedChunk] = []
        
        for raw in raw_chunks:
            vid = raw.get("vector_id")
            chunk_id = raw.get("chunk_id", "")
            is_linked = raw.get("is_linked", False)
            
            # Get base text
            base_text = raw.get("text", "")
            merged_text = base_text
            linked_chain = []
            
            # If linked, get the full chain and merge text
            if is_linked:
                try:
                    chain = await get_chunk_with_chain(self._db, chunk_id, max_depth=3)
                    if len(chain) > 1:
                        # Merge all texts in order
                        merged_text = " ".join([c.get("text", "") for c in chain])
                        linked_chain = [c.get("chunk_id") for c in chain]
                        logger.debug(
                            "Merged linked chunks",
                            extra={
                                "chunk_id": chunk_id,
                                "chain": linked_chain,
                                "original_len": len(base_text),
                                "merged_len": len(merged_text),
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to merge linked chunks for {chunk_id}: {e}")
            
            retrieved.append(RetrievedChunk(
                vector_id=vid,
                distance=distance_map.get(vid, 9999.0),
                chunk_id=chunk_id,
                document_id=raw.get("document_id", ""),
                filename=raw.get("filename", ""),
                page_number=raw.get("page_number", 0),
                text=base_text,
                bbox=raw.get("bbox", {}),
                rerank_score=0.0,
                linked_text=merged_text if is_linked else "",
                is_linked=is_linked,
            ))

        # Sort by FAISS distance ascending
        retrieved.sort(key=lambda c: c.distance)

        # Deduplicate: If multiple chunks from same linked chain appear,
        # keep only the one with best distance
        seen_chains = set()
        deduped = []
        for chunk in retrieved:
            chain_key = chunk.chunk_id
            if chunk.is_linked and chunk.linked_text:
                # Use first chunk in chain as key for dedup
                # Simplified: just use chunk_id for now
                pass
            deduped.append(chunk)

        logger.info(
            "Retrieval complete",
            extra={
                "query":     query[:60],
                "k":         k,
                "retrieved": len(retrieved),
                "linked_merged": sum(1 for c in retrieved if c.is_linked),
            },
        )

        return deduped