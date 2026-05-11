"""
db/queries.py
=============
All reusable MongoDB query functions.

Changes:
    Added get_chunk_with_chain() — retrieves chunk and all linked siblings.
"""

from typing import Optional  # ← ADD THIS LINE
from motor.motor_asyncio import AsyncIOMotorDatabase
from core.logger import get_logger
from core.exceptions import DatabaseError
from db.models import DocumentRecord, ChunkRecord, ResultRecord

logger = get_logger(__name__)


# ── Documents ─────────────────────────────────────────────────────────────────

async def insert_document(
    db: AsyncIOMotorDatabase,
    record: DocumentRecord,
) -> str:
    try:
        await db["documents"].replace_one(
            {"_id": record.document_id},
            record.to_mongo(),
            upsert=True,
        )
        logger.info("Document record saved", extra={"document_id": record.document_id})
        return record.document_id
    except Exception as exc:
        raise DatabaseError("insert_document", str(exc)) from exc


async def get_all_documents(db: AsyncIOMotorDatabase) -> list[dict]:
    try:
        cursor = db["documents"].find({}, {"_id": 0})
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_all_documents", str(exc)) from exc


async def get_document_by_id(
    db: AsyncIOMotorDatabase,
    document_id: str,
) -> dict | None:
    try:
        return await db["documents"].find_one({"_id": document_id}, {"_id": 0})
    except Exception as exc:
        raise DatabaseError("get_document_by_id", str(exc)) from exc


async def count_documents(db: AsyncIOMotorDatabase) -> int:
    try:
        return await db["documents"].count_documents({})
    except Exception as exc:
        raise DatabaseError("count_documents", str(exc)) from exc


# ── Chunks ────────────────────────────────────────────────────────────────────

async def insert_chunks(
    db: AsyncIOMotorDatabase,
    chunks: list[ChunkRecord],
) -> int:
    if not chunks:
        return 0
    try:
        docs = [c.to_mongo() for c in chunks]
        result = await db["chunks"].insert_many(docs, ordered=False)
        logger.info("Chunks inserted", extra={"count": len(result.inserted_ids)})
        return len(result.inserted_ids)
    except Exception as exc:
        raise DatabaseError("insert_chunks", str(exc)) from exc


async def get_chunk_by_vector_id(
    db: AsyncIOMotorDatabase,
    vector_id: int,
) -> dict | None:
    try:
        return await db["chunks"].find_one({"vector_id": vector_id})
    except Exception as exc:
        raise DatabaseError("get_chunk_by_vector_id", str(exc)) from exc


async def get_chunks_by_vector_ids(
    db: AsyncIOMotorDatabase,
    vector_ids: list[int],
) -> list[dict]:
    try:
        cursor = db["chunks"].find({"vector_id": {"$in": vector_ids}})
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_chunks_by_vector_ids", str(exc)) from exc


async def get_chunk_by_id(
    db: AsyncIOMotorDatabase,
    chunk_id: str,
) -> dict | None:
    """Get a single chunk by its chunk_id."""
    try:
        return await db["chunks"].find_one({"_id": chunk_id})
    except Exception as exc:
        raise DatabaseError("get_chunk_by_id", str(exc)) from exc


async def get_chunk_with_chain(
    db: AsyncIOMotorDatabase,
    chunk_id: str,
    max_depth: int = 3,
) -> list[dict]:
    """
    Retrieve a chunk and all linked chunks (prev/next chain).
    
    Follows next_chunk_id and prev_chunk_id pointers to reconstruct
    the complete original text for chunks that were split across boundaries.
    
    Args:
        db: Database connection
        chunk_id: Starting chunk ID
        max_depth: Maximum number of chunks to follow in each direction
    
    Returns:
        List of chunk documents in original order (oldest to newest)
    """
    try:
        # Get the starting chunk
        start_chunk = await get_chunk_by_id(db, chunk_id)
        if not start_chunk:
            return []
        
        # If not linked, just return this chunk
        if not start_chunk.get("is_linked"):
            return [start_chunk]
        
        # Go to the first chunk in the chain
        chain = []
        current = start_chunk
        depth = 0
        
        # Walk backwards to find first chunk
        while current.get("prev_chunk_id") and depth < max_depth:
            prev = await get_chunk_by_id(db, current["prev_chunk_id"])
            if not prev:
                break
            current = prev
            depth += 1
        
        # Walk forward collecting all chunks
        depth = 0
        while current and depth < max_depth:
            chain.append(current)
            if current.get("next_chunk_id"):
                current = await get_chunk_by_id(db, current["next_chunk_id"])
                depth += 1
            else:
                break
        
        return chain
        
    except Exception as exc:
        raise DatabaseError("get_chunk_with_chain", str(exc)) from exc


async def get_chunks_by_document_and_range(
    db: AsyncIOMotorDatabase,
    document_id: str,
    start_index: int,
    end_index: int,
) -> list[dict]:
    """Get chunks from a specific document within a range of chunk indices."""
    try:
        cursor = db["chunks"].find({
            "document_id": document_id,
            "chunk_index": {"$gte": start_index, "$lt": end_index}
        }).sort("chunk_index", 1)
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_chunks_by_document_and_range", str(exc)) from exc


async def delete_chunks_by_document(
    db: AsyncIOMotorDatabase,
    document_id: str,
) -> int:
    try:
        result = await db["chunks"].delete_many({"document_id": document_id})
        logger.info(
            "Chunks deleted",
            extra={"document_id": document_id, "count": result.deleted_count},
        )
        return result.deleted_count
    except Exception as exc:
        raise DatabaseError("delete_chunks_by_document", str(exc)) from exc


async def update_chunk_links(
    db: AsyncIOMotorDatabase,
    chunk_id: str,
    next_chunk_id: Optional[str] = None,
    prev_chunk_id: Optional[str] = None,
    is_linked: bool = True,
) -> bool:
    """Update link fields for a chunk."""
    try:
        update_data = {}
        if next_chunk_id is not None:
            update_data["next_chunk_id"] = next_chunk_id
        if prev_chunk_id is not None:
            update_data["prev_chunk_id"] = prev_chunk_id
        if is_linked is not None:
            update_data["is_linked"] = is_linked
        
        if not update_data:
            return True
        
        result = await db["chunks"].update_one(
            {"_id": chunk_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except Exception as exc:
        raise DatabaseError("update_chunk_links", str(exc)) from exc


# ── Results ───────────────────────────────────────────────────────────────────

async def insert_result(
    db: AsyncIOMotorDatabase,
    record: ResultRecord,
) -> str:
    try:
        doc = record.to_mongo()
        await db["results"].insert_one(doc)
        logger.info(
            "Result saved",
            extra={"query_hash": record.query_hash, "spans": len(record.spans)},
        )
        return str(doc["_id"])
    except Exception as exc:
        raise DatabaseError("insert_result", str(exc)) from exc


async def get_recent_results(
    db: AsyncIOMotorDatabase,
    limit: int = 20,
) -> list[dict]:
    try:
        cursor = db["results"].find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_recent_results", str(exc)) from exc