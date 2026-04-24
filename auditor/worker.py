"""
auditor/worker.py
=================
Single QA worker — processes one retrieved chunk.

Each worker:
    1. Takes a (question, RetrievedChunk) pair
    2. Runs QAModel.predict() to extract a span
    3. Returns a WorkerResult enriched with source metadata

The dispatcher runs multiple workers concurrently using
asyncio + ProcessPoolExecutor (CoW memory sharing).

Note: QAModel inference is CPU-bound, so we offload it to
a thread pool executor to avoid blocking the async event loop.
"""

import asyncio
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
        qa_result    — QAResult from the model (has span, scores)
        chunk        — the RetrievedChunk that was processed
        success      — False if the worker threw an exception
        error        — error message if success is False
    """
    qa_result: QAResult | None
    chunk: RetrievedChunk
    success: bool
    error: str = ""


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

    def __init__(self, qa_model: QAModel, executor: ThreadPoolExecutor | None = None):
        self._model = qa_model
        self._executor = executor  # shared thread pool from dispatcher

    async def run(self, question: str, chunk: RetrievedChunk) -> WorkerResult:
        """
        Run QA inference on one chunk asynchronously.

        Offloads CPU-bound model inference to a thread pool executor
        so it doesn't block the asyncio event loop.

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
                    "filename": chunk.filename,
                    "page": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                },
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
                "Worker QA failed",
                extra={
                    "chunk_id": chunk.chunk_id,
                    "error": str(exc),
                },
            )
            return WorkerResult(
                qa_result=None,
                chunk=chunk,
                success=False,
                error=str(exc),
            )

        except Exception as exc:
            logger.error(
                "Worker unexpected error",
                extra={"chunk_id": chunk.chunk_id, "error": str(exc)},
            )
            return WorkerResult(
                qa_result=None,
                chunk=chunk,
                success=False,
                error=str(exc),
            )