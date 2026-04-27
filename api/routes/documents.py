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
    from pathlib import Path
    from core.config import settings
    
    # Get document to retrieve original filename
    doc = await db["documents"].find_one({"_id": document_id})
    
    # Delete chunks
    await delete_chunks_by_document(db, document_id)
    
    # Delete document record
    await db["documents"].delete_one({"_id": document_id})
    
    # Delete the file using original filename from MongoDB
    if doc and "filename" in doc:
        file_path = Path(settings.upload_dir) / doc["filename"]
        if file_path.exists():
            file_path.unlink()
            logger.info(
                "Document file deleted",
                extra={"document_id": document_id, "filename": doc["filename"]}
            )

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
    Handles both new (original filename) and old (hash-based) storage formats.
    """
    from pathlib import Path
    from fastapi.responses import FileResponse
    from core.config import settings
    
    # Check if document exists in MongoDB
    doc = await db["documents"].find_one({"_id": document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Use the original filename stored in MongoDB
    original_filename = doc.get("filename")
    if not original_filename:
        raise HTTPException(status_code=500, detail="Filename not found in database record")
    
    file_path = Path(settings.upload_dir) / original_filename
    
    # Backward compatibility: if file doesn't exist, try hash-based name
    if not file_path.exists():
        hash_based_filename = f"{document_id}.pdf"
        hash_based_path = Path(settings.upload_dir) / hash_based_filename
        if hash_based_path.exists():
            logger.info(
                "Found file with hash-based name (legacy), migrating to original filename",
                extra={"document_id": document_id, "legacy_name": hash_based_filename}
            )
            # Migrate: rename old hash-based file to original filename
            try:
                hash_based_path.rename(file_path)
            except Exception as exc:
                logger.warning(
                    "Failed to migrate file to original filename",
                    extra={"document_id": document_id, "error": str(exc)}
                )
                # Use the hash-based file anyway
                file_path = hash_based_path
        else:
            logger.error(
                "PDF file not found",
                extra={"document_id": document_id, "filename": original_filename, "path": str(file_path)}
            )
            raise HTTPException(status_code=404, detail="PDF file not found on disk")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")
    
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=original_filename,
    )