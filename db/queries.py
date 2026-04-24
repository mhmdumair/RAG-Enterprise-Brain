"""
db/queries.py
=============
All reusable MongoDB query functions.
No other module writes raw pymongo/motor queries — they call these.

Every function accepts a database handle as its first argument
so it can be injected and mocked cleanly in tests.
"""

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
    """
    Insert a DocumentRecord into the documents collection.
    If the document_id already exists, replace it (re-ingestion).

    Returns:
        The document_id of the inserted record.
    """
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
    """
    Return all ingested document records.
    Used by GET /documents endpoint.
    """
    try:
        cursor = db["documents"].find({}, {"_id": 0})
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_all_documents", str(exc)) from exc


async def get_document_by_id(
    db: AsyncIOMotorDatabase,
    document_id: str,
) -> dict | None:
    """Return a single document record or None if not found."""
    try:
        return await db["documents"].find_one({"_id": document_id}, {"_id": 0})
    except Exception as exc:
        raise DatabaseError("get_document_by_id", str(exc)) from exc


async def count_documents(db: AsyncIOMotorDatabase) -> int:
    """Return the total number of ingested documents."""
    try:
        return await db["documents"].count_documents({})
    except Exception as exc:
        raise DatabaseError("count_documents", str(exc)) from exc


# ── Chunks ────────────────────────────────────────────────────────────────────

async def insert_chunks(
    db: AsyncIOMotorDatabase,
    chunks: list[ChunkRecord],
) -> int:
    """
    Bulk insert a list of ChunkRecords into the chunks collection.
    Uses ordered=False so partial failures don't block the rest.

    Returns:
        Number of chunks successfully inserted.
    """
    if not chunks:
        return 0
    try:
        docs = [c.to_mongo() for c in chunks]
        result = await db["chunks"].insert_many(docs, ordered=False)
        logger.info(
            "Chunks inserted",
            extra={"count": len(result.inserted_ids)},
        )
        return len(result.inserted_ids)
    except Exception as exc:
        raise DatabaseError("insert_chunks", str(exc)) from exc


async def get_chunk_by_vector_id(
    db: AsyncIOMotorDatabase,
    vector_id: int,
) -> dict | None:
    """
    Look up a chunk by its FAISS vector index position.
    This is the critical bridge between FAISS search results and MongoDB.

    Returns:
        The chunk document or None if not found.
    """
    try:
        return await db["chunks"].find_one({"vector_id": vector_id})
    except Exception as exc:
        raise DatabaseError("get_chunk_by_vector_id", str(exc)) from exc


async def get_chunks_by_vector_ids(
    db: AsyncIOMotorDatabase,
    vector_ids: list[int],
) -> list[dict]:
    """
    Batch lookup — return chunks for a list of FAISS vector IDs.
    Used by the retriever after ANN search returns top-K IDs.
    """
    try:
        cursor = db["chunks"].find({"vector_id": {"$in": vector_ids}})
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_chunks_by_vector_ids", str(exc)) from exc


async def get_chunks_by_document_and_range(
    db: AsyncIOMotorDatabase,
    document_id: str,
    start_index: int,
    end_index: int,
) -> list[dict]:
    """
    Get chunks from a specific document within a range of chunk indices.
    Used for context stitching.
    
    Args:
        db: Database connection
        document_id: The document to query
        start_index: Starting chunk index (inclusive)
        end_index: Ending chunk index (exclusive)
    
    Returns:
        List of chunks sorted by chunk_index
    """
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
    """
    Delete all chunks belonging to a document.
    Called when re-ingesting an existing PDF.

    Returns:
        Number of chunks deleted.
    """
    try:
        result = await db["chunks"].delete_many({"document_id": document_id})
        logger.info(
            "Chunks deleted",
            extra={"document_id": document_id, "count": result.deleted_count},
        )
        return result.deleted_count
    except Exception as exc:
        raise DatabaseError("delete_chunks_by_document", str(exc)) from exc


# ── Results ───────────────────────────────────────────────────────────────────

async def insert_result(
    db: AsyncIOMotorDatabase,
    record: ResultRecord,
) -> str:
    """
    Persist an audit result to the results collection.

    Returns:
        The MongoDB _id of the inserted result.
    """
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
    """
    Return the most recent audit results ordered by created_at descending.
    """
    try:
        cursor = db["results"].find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=None)
    except Exception as exc:
        raise DatabaseError("get_recent_results", str(exc)) from exc