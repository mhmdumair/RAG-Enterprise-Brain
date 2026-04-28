"""
brain/store.py
==============
MongoDB persistence for chunk metadata.

After FAISS assigns integer vector IDs to each chunk,
this module saves the mapping:
    vector_id (int) → filename, page_number, bbox, text, document_id

This bridge is what makes results attributable to a source document.

Changes from previous version:
    ChunkRecord construction now passes is_fragment and sentence_count.
    All method signatures are identical — nothing outside this file changes.

Flow:
    list[TextChunk] + list[vector_id] → ChunkRecord → MongoDB chunks collection
"""

from motor.motor_asyncio import AsyncIOMotorDatabase

from core.logger import get_logger
from core.exceptions import DatabaseError
from db.models import DocumentRecord, ChunkRecord
from db.queries import (
    insert_document,
    insert_chunks,
    delete_chunks_by_document,
)
from brain.parser import ParsedDocument
from brain.chunker import TextChunk

logger = get_logger(__name__)


class ChunkStore:
    """
    Saves ingestion results (document record + chunk records) to MongoDB.

    Public interface is identical to the previous version.
    brain/pipeline.py calls this and is unaffected.

    Usage:
        store = ChunkStore(db)
        await store.save(parsed_doc, chunks, vector_ids)
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self._db = db

    async def save(
        self,
        parsed_doc: ParsedDocument,
        chunks:     list[TextChunk],
        vector_ids: list[int],
    ) -> None:
        """
        Persist the document record and all chunk records to MongoDB.

        Steps:
            1. Delete existing chunks if document was previously ingested
            2. Insert / upsert the DocumentRecord
            3. Bulk insert all ChunkRecords with their vector_ids

        Args:
            parsed_doc  — output of PDFParser (filename, doc_id, pages)
            chunks      — output of TextChunker (text, bbox, page_number,
                          is_fragment, sentence_count)
            vector_ids  — output of FAISSIndex.add() — same length as chunks

        Raises:
            ValueError    — if chunks and vector_ids lengths do not match
            DatabaseError — if any MongoDB operation fails
        """
        if len(chunks) != len(vector_ids):
            raise ValueError(
                f"Chunks ({len(chunks)}) and vector_ids "
                f"({len(vector_ids)}) must be the same length."
            )

        logger.info(
            "Saving ingestion to MongoDB",
            extra={
                "document_id": parsed_doc.document_id,
                "doc_file":    parsed_doc.filename,   # ← renamed from "filename"
                "chunks":      len(chunks),
                "fragments":   sum(1 for c in chunks if c.is_fragment),
            },
        )

        # ── Step 1: Remove old chunks for this document (re-ingestion) ────
        deleted = await delete_chunks_by_document(
            self._db,
            parsed_doc.document_id,
        )
        if deleted > 0:
            logger.info(
                "Removed old chunks for re-ingestion",
                extra={
                    "document_id": parsed_doc.document_id,
                    "deleted":     deleted,
                },
            )

        # ── Step 2: Upsert document record ────────────────────────────────
        doc_record = DocumentRecord(
            document_id=parsed_doc.document_id,
            filename=parsed_doc.filename,
            total_pages=len(parsed_doc.pages),
            total_chunks=len(chunks),
            file_size_mb=parsed_doc.file_size_mb,
            status="ingested",
        )
        await insert_document(self._db, doc_record)

        # ── Step 3: Build and bulk insert chunk records ───────────────────
        chunk_records = [
            ChunkRecord(
                chunk_id=chunk.chunk_id,
                vector_id=vector_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                bbox=chunk.bbox,
                is_fragment=chunk.is_fragment,        # ← new field
                sentence_count=chunk.sentence_count,  # ← new field
            )
            for chunk, vector_id in zip(chunks, vector_ids)
        ]

        inserted = await insert_chunks(self._db, chunk_records)

        logger.info(
            "Ingestion saved to MongoDB",
            extra={
                "document_id":     parsed_doc.document_id,
                "chunks_inserted": inserted,
            },
        )