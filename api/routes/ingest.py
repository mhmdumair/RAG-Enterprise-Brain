"""
api/routes/ingest.py
====================
POST /ingest — upload and ingest a PDF file.

Flow:
    1. Validate: file type must be PDF
    2. Validate: file size must be within limit
    3. Save to storage/uploads/
    4. Run IngestionPipeline.run()
    5. Return IngestResponse

Accepts: multipart/form-data with a 'file' field.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import FileTooLargeError, TooManyPDFsError, PDFParseError
from brain.pipeline import IngestionPipeline
from api.schemas import IngestResponse, ErrorResponse
from api.dependencies import get_db, get_pipeline

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a PDF document",
    tags=["Ingestion"],
    status_code=201,
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest."),
    db: AsyncIOMotorDatabase = Depends(get_db),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    """
    Upload and ingest a PDF file into the Enterprise Brain.

    - File must be a PDF (checked by content-type and extension).
    - File must be within the size limit (default 50MB).
    - Maximum 10 PDFs can be ingested (re-ingesting existing file replaces it).
    - Returns ingestion summary with chunk and vector counts.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail=ErrorResponse(
                error="UnsupportedMediaType",
                message="Only PDF files are accepted.",
                details={"filename": filename}
            ).model_dump(),
        )
    
    logger.info(
        "Ingest request received",
        extra={
            "pdf_filename": filename,  # Changed from 'filename' to avoid KeyError
            "content_type": file.content_type,
        }
    )

    # ── Read and size-check ───────────────────────────────────────────────────
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > settings.max_file_size_mb:
        raise FileTooLargeError(filename, size_mb, settings.max_file_size_mb)

    # ── Save to disk ──────────────────────────────────────────────────────────
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / filename
    
    try:
        with open(upload_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        logger.error(
            "Failed to save uploaded file",
            extra={"pdf_filename": filename, "error": str(exc)}
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="FileSaveError",
                message=f"Failed to save uploaded file: {exc}",
                details={"filename": filename}
            ).model_dump(),
        )

    logger.info(
        "File uploaded",
        extra={"pdf_filename": filename, "size_mb": round(size_mb, 2)},
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = await pipeline.run(upload_path)
    except Exception as exc:
        # Clean up temp file if pipeline fails
        if upload_path.exists():
            upload_path.unlink()
        raise
    
    # Clean up temp file after successful ingestion
    if upload_path.exists():
        upload_path.unlink()
        logger.debug(
            "Uploaded file cleaned up",
            extra={"pdf_filename": filename, "path": str(upload_path)}
        )

    logger.info(
        "Ingest completed successfully",
        extra={
            "pdf_filename": filename,
            "document_id": result.document_id,
            "chunks": result.total_chunks,
        }
    )

    return IngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        total_pages=result.total_pages,
        total_chunks=result.total_chunks,
        total_vectors=result.total_vectors,
        message=f"{result.filename} ingested successfully.",
    )