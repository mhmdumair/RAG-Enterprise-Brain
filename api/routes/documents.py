"""
api/routes/documents.py
=======================
GET /documents — list all ingested documents.

Returns metadata for every PDF that has been successfully ingested:
  - document_id, filename, total_pages, total_chunks, file_size_mb,
    status, created_at
"""

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.logger import get_logger
from db.queries import get_all_documents
from api.schemas import DocumentListResponse, DocumentSchema
from api.dependencies import get_db

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    tags=["Documents"],
)
async def list_documents(
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Return all ingested PDF documents with their metadata.

    Returns an empty list if no documents have been ingested yet.
    """
    raw_docs = await get_all_documents(db)

    documents = [
        DocumentSchema(
            document_id=doc["document_id"],
            filename=doc["filename"],
            total_pages=doc["total_pages"],
            total_chunks=doc["total_chunks"],
            file_size_mb=doc["file_size_mb"],
            status=doc["status"],
            created_at=doc["created_at"],
        )
        for doc in raw_docs
    ]

    logger.info("Documents listed", extra={"count": len(documents)})

    return DocumentListResponse(
        documents=documents,
        total=len(documents),
    )