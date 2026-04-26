"""
api/routes/documents.py
=======================
GET /documents — list all ingested documents.

Returns metadata for every PDF that has been successfully ingested:
  - document_id, filename, total_pages, total_chunks, file_size_mb,
    status, created_at
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
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

@router.delete(
    "/documents/{document_id}",
    summary="Delete an ingested document",
    tags=["Documents"],
    status_code=204,
)
async def delete_document(
    document_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    from db.queries import delete_chunks_by_document
    from motor.motor_asyncio import AsyncIOMotorDatabase
    await delete_chunks_by_document(db, document_id)
    await db["documents"].delete_one({"_id": document_id})

@router.get(
    "/documents/{document_id}/file",
    summary="Get the original PDF file",
    tags=["Documents"],
    response_class=FileResponse,
)
async def get_document_file(
    document_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Return the original PDF file for the given document ID.
    """
    from pathlib import Path
    from fastapi.responses import FileResponse
    from core.config import settings
    
    # Check if document exists
    doc = await db["documents"].find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = Path(settings.upload_dir) / f"{document_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=doc["filename"],
    )