"""
brain/pipeline.py
=================
Ingestion pipeline orchestrator — Boundary 1 entry point.

Ties together all brain components in the correct sequence:
    parse → chunk → embed → index → store

This is the single function the API calls when a PDF is uploaded.
All other brain modules are internal implementation details.

Flow:
    PDF path
      └─ PDFParser.parse()           → ParsedDocument
      └─ TextChunker.chunk()         → list[TextChunk]
      └─ Embedder.embed()            → np.ndarray (N x 384)
      └─ FAISSIndex.add()            → list[vector_id]
      └─ ChunkStore.save()           → MongoDB persisted
      └─ FAISSIndex.save()           → disk persisted
      └─ PipelineResult              → returned to caller
"""

from dataclasses import dataclass
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import IngestionError, TooManyPDFsError
from core.utils import make_document_id
from db.queries import count_documents, get_document_by_id
from brain.parser import PDFParser
from brain.chunker import TextChunker
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from brain.store import ChunkStore

logger = get_logger(__name__)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Returned by IngestionPipeline.run() after successful ingestion.

    Fields:
        document_id   — deterministic ID of the ingested document
        filename      — original PDF filename
        total_pages   — number of pages processed
        total_chunks  — number of chunks created and indexed
        total_vectors — total vectors now in the FAISS index
    """
    document_id: str
    filename: str
    total_pages: int
    total_chunks: int
    total_vectors: int


# ── Pipeline ──────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates the full PDF ingestion flow.

    Components are injected so they can be shared across requests
    (embedder and FAISS index are expensive to reload).

    Usage:
        pipeline = IngestionPipeline(db, faiss_index)
        result = await pipeline.run(Path("document.pdf"))
    """

    def __init__(self, db: AsyncIOMotorDatabase, faiss_index: FAISSIndex):
        self._db = db
        self._faiss = faiss_index
        self._parser = PDFParser()
        self._chunker = TextChunker()
        self._embedder = Embedder()
        self._store = ChunkStore(db)

    async def run(self, file_path: Path) -> PipelineResult:
        """
        Run the full ingestion pipeline for one PDF file.

        Args:
            file_path — absolute path to the uploaded PDF

        Returns:
            PipelineResult with ingestion summary.

        Raises:
            TooManyPDFsError — if max_pdfs limit is reached
            IngestionError   — if any pipeline stage fails
            FileTooLargeError, PDFParseError — from parser
        """
        file_path = Path(file_path)
        filename = file_path.name
        document_id = make_document_id(filename)

        logger.info(
            "Pipeline started",
            extra={"pdf_filename": filename, "document_id": document_id},  # Changed 'filename' to 'pdf_filename'
        )

        # ── Guard: PDF count limit ────────────────────────────────────────────
        existing_count = await count_documents(self._db)
        is_reingest = await get_document_by_id(self._db, document_id) is not None

        if not is_reingest and existing_count >= settings.max_pdfs:
            raise TooManyPDFsError(existing_count, settings.max_pdfs)

        # ── Stage 1: Parse ────────────────────────────────────────────────────
        try:
            parsed_doc = self._parser.parse(file_path, document_id)
        except Exception as exc:
            raise IngestionError(
                f"Parse stage failed for '{filename}': {exc}"
            ) from exc

        # ── Stage 2: Chunk ────────────────────────────────────────────────────
        try:
            chunks = self._chunker.chunk(parsed_doc)
        except Exception as exc:
            raise IngestionError(
                f"Chunk stage failed for '{filename}': {exc}"
            ) from exc

        if not chunks:
            raise IngestionError(
                f"No chunks produced from '{filename}' — document may be empty."
            )

        # ── Stage 3: Embed ────────────────────────────────────────────────────
        try:
            vectors = self._embedder.embed(chunks)
        except Exception as exc:
            raise IngestionError(
                f"Embed stage failed for '{filename}': {exc}"
            ) from exc

        # ── Stage 4: Index ────────────────────────────────────────────────────
        try:
            if self._faiss.is_ready:
                vector_ids = self._faiss.add(vectors)
            else:
                self._faiss.build(vectors)
                vector_ids = list(range(len(chunks)))
        except Exception as exc:
            raise IngestionError(
                f"Index stage failed for '{filename}': {exc}"
            ) from exc

        # ── Stage 5: Store ────────────────────────────────────────────────────
        try:
            await self._store.save(parsed_doc, chunks, vector_ids)
        except Exception as exc:
            raise IngestionError(
                f"Store stage failed for '{filename}': {exc}"
            ) from exc

        # ── Persist index to disk ─────────────────────────────────────────────
        try:
            self._faiss.save()
        except Exception as exc:
            logger.warning(
                "FAISS index save failed — index is in memory but not persisted",
                extra={"error": str(exc)},
            )

        result = PipelineResult(
            document_id=document_id,
            filename=filename,
            total_pages=len(parsed_doc.pages),
            total_chunks=len(chunks),
            total_vectors=self._faiss.total_vectors,
        )

        logger.info(
            "Pipeline complete",
            extra={
                "pdf_filename": filename,  # Changed 'filename' to 'pdf_filename'
                "pages": result.total_pages,
                "chunks": result.total_chunks,
                "total_vectors": result.total_vectors,
            },
        )

        return result