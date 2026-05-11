"""
auditor/worker.py
=================
Single QA worker — processes one retrieved chunk.

Each worker:
    1. Takes a (question, RetrievedChunk) pair
    2. Runs QAModel.predict() to extract a span
    3. Returns a WorkerResult enriched with source metadata

The dispatcher runs multiple workers concurrently using asyncio.gather.
CPU-bound QA inference is offloaded to a thread pool executor.
"""

import asyncio
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from core.logger import get_logger
from core.exceptions import QAModelError
from auditor.qa_model import QAModel, QAResult
from auditor.retriever import RetrievedChunk

logger = get_logger(__name__)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class WorkerResult:
    """
    Output of a single worker run.

    Fields:
        qa_result — QAResult from the model (has span, scores)
        chunk     — the RetrievedChunk that was processed
        success   — False if the worker threw an exception
        error     — error message if success is False
    """
    qa_result: QAResult | None
    chunk:     RetrievedChunk
    success:   bool
    error:     str = ""


# ── Worker ────────────────────────────────────────────────────────────────────

class QAWorker:
    """
    Runs extractive QA on a single chunk.

    The QAModel is injected so it can be shared across workers
    (model weights are read-only after loading — safe to share).

    Usage:
        worker = QAWorker(qa_model)
        result = await worker.run(question, chunk)
    """

    def __init__(
        self,
        qa_model:  QAModel,
        executor:  ThreadPoolExecutor | None = None,
    ):
        self._model    = qa_model
        self._executor = executor

    async def run(
        self,
        question: str,
        chunk:    RetrievedChunk,
    ) -> WorkerResult:
        """
        Run QA inference on one chunk asynchronously.

        Offloads CPU-bound model inference to a thread pool executor
        so it does not block the asyncio event loop.

        Args:
            question — the audit query string
            chunk    — the RetrievedChunk to run QA against

        Returns:
            WorkerResult with QAResult and source chunk.
        """
        loop = asyncio.get_event_loop()

        try:
            logger.debug(
                "Worker running",
                extra={
                    "doc_file":   chunk.filename,
                    "page":       chunk.page_number,
                    "chunk_id":   chunk.chunk_id,
                    "text_len":   len(chunk.text),
                },
            )

            # Validate chunk text before sending to model
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has empty text — skipping."
                )

            # Run CPU-bound inference in thread pool
            qa_result = await loop.run_in_executor(
                self._executor,
                self._model.predict,
                question,
                chunk.text,
            )

            return WorkerResult(
                qa_result=qa_result,
                chunk=chunk,
                success=True,
            )

        except QAModelError as exc:
            logger.warning(
                "Worker QA model error",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "error":    str(exc),
                },
            )
            return WorkerResult(
                qa_result=None,
                chunk=chunk,
                success=False,
                error=str(exc),
            )

        except Exception as exc:
            # Log full traceback so we can see the real error
            tb = traceback.format_exc()
            logger.error(
                "Worker unexpected error",
                extra={
                    "chunk_id":  chunk.chunk_id,
                    "doc_file":  chunk.filename,
                    "page":      chunk.page_number,
                    "error":     str(exc),
                    "type":      type(exc).__name__,
                    "traceback": tb,
                },
            )
            return WorkerResult(
                qa_result=None,
                chunk=chunk,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )