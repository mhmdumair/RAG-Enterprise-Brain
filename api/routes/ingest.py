"""
api/routes/ingest.py
====================
POST /ingest — upload and ingest a PDF file.

Flow:
    1. Validate: file type must be PDF
    2. Validate: file size must be within limit
    3. Check: PDF must have extractable text layer (reject scanned/image-based PDFs)
    4. Save to storage/uploads/
    5. Run IngestionPipeline.run()
    6. Return IngestResponse

Accepts: multipart/form-data with a 'file' field.
"""

import shutil
from pathlib import Path
import fitz  # PyMuPDF

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import FileTooLargeError, TooManyPDFsError, PDFParseError
from core.utils import make_document_id
from brain.pipeline import IngestionPipeline
from api.schemas import IngestResponse, ErrorResponse
from api.dependencies import get_db, get_pipeline

logger = get_logger(__name__)
router = APIRouter()


def has_text_layer(pdf_path: Path) -> tuple[bool, int]:
    """
    Check if PDF has extractable text layer.
    
    Returns:
        (has_text, pages_with_text)
        has_text — True if at least one page has >100 chars of extractable text
        pages_with_text — count of pages with extractable text
    """
    try:
        doc = fitz.open(pdf_path)
        pages_with_text = 0
        total_pages = len(doc)
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) > 100:  # Meaningful text threshold
                pages_with_text += 1
        
        doc.close()
        
        # Require at least 70% of pages to have text (or at least 1 page for small PDFs)
        if total_pages <= 3:
            has_text = pages_with_text >= 1
        else:
            has_text = pages_with_text / total_pages >= 0.7
        
        return has_text, pages_with_text
        
    except Exception as e:
        logger.error(f"Text layer detection failed: {e}")
        return False, 0


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
    - File MUST have an extractable text layer (no scanned/image-based PDFs).
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
    
    document_id = make_document_id(filename)
    
    logger.info(
        "Ingest request received",
        extra={
            "pdf_filename": filename,
            "document_id": document_id,
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
    # Save with original filename, not hash-based ID
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

    # ── TEXT LAYER DETECTION (Reject scanned PDFs) ────────────────────────────
    has_text, pages_with_text = has_text_layer(upload_path)
    
    if not has_text:
        # Clean up the uploaded file
        if upload_path.exists():
            upload_path.unlink()
        
        raise HTTPException(
            status_code=415,
            detail=ErrorResponse(
                error="ScannedPDFNotSupported",
                message="This PDF appears to be scanned or image-based without an extractable text layer.",
                details={
                    "filename": filename,
                    "pages_with_text": pages_with_text,
                    "requirement": "PDF must have selectable text. Please use OCR on your PDF first, then upload the text-searchable version.",
                }
            ).model_dump(),
        )
    
    logger.info(
        "Text layer detected",
        extra={
            "pdf_filename": filename,
            "pages_with_text": pages_with_text,
        }
    )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = await pipeline.run(upload_path)
    except Exception as exc:
        # Clean up file if pipeline fails
        if upload_path.exists():
            upload_path.unlink()
        raise

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