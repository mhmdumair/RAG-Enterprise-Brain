"""
auditor/dispatcher.py
=====================
Parallel QA dispatcher — Boundary 2 entry point.

Orchestrates the full audit query flow:
    1. Retrieve top-K chunks via ANN search
    2. Run QA concurrently on all chunks (asyncio.gather + thread pool)
    3. Apply abstention filter (S_span > S_null + τ_ans)
    4. If all fail → trigger RAKE fallback → retry retrieval once
    5. Deduplicate spans (SHA-256)
    6. Return ranked, attributed VerifiedAnswer list

This is the single function the API route calls.
"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import re

from motor.motor_asyncio import AsyncIOMotorDatabase
import asyncio

from core.config import settings
from core.logger import get_logger
from core.exceptions import NoAnswerFoundError
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from auditor.retriever import Retriever, RetrievedChunk
from auditor.qa_model import QAModel, QAResult
from auditor.abstention import AbstentionFilter
from auditor.rake_fallback import RAKEFallback
from auditor.deduplicator import Deduplicator
from auditor.worker import QAWorker, WorkerResult

logger = get_logger(__name__)


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class VerifiedAnswer:
    """
    A single verified, attributed answer span.
    This is what the API returns to the frontend.

    Fields:
        text         — the extracted answer string
        span_score   — confidence score (higher = more confident)
        null_score   — no-answer score (for reference)
        filename     — source PDF filename
        page_number  — source page (1-based)
        bbox         — normalized BBox for frontend highlighting
        chunk_text   — full chunk context (for display)
        span_hash    — SHA-256 of normalized answer (dedup fingerprint)
        rake_used    — True if this answer came from RAKE fallback retry
    """
    text: str
    span_score: float
    null_score: float
    filename: str
    page_number: int
    bbox: dict
    chunk_text: str
    span_hash: str
    rake_used: bool = False


@dataclass
class DispatchResult:
    """
    Full result returned by the dispatcher.

    Fields:
        answers              — ranked list of VerifiedAnswer
        query                — original query string
        rake_used            — True if RAKE fallback was triggered
        total_chunks_searched— how many chunks were evaluated
    """
    answers: list[VerifiedAnswer]
    query: str
    rake_used: bool
    total_chunks_searched: int


# ── Dispatcher ────────────────────────────────────────────────────────────────

class AuditDispatcher:
    """
    Parallel audit dispatcher.

    Components are injected and shared across requests — the QA model
    and embedder are expensive to reload. The thread pool is also
    shared for efficient CPU utilization.

    Usage:
        dispatcher = AuditDispatcher(embedder, faiss_index, db)
        result = await dispatcher.dispatch("What is the warranty period?")
    """

    def __init__(
        self,
        embedder: Embedder,
        faiss_index: FAISSIndex,
        db: AsyncIOMotorDatabase,
    ):
        self._retriever = Retriever(embedder, faiss_index, db)
        self._qa_model = QAModel()
        self._abstention = AbstentionFilter()
        self._rake = RAKEFallback()
        self._deduplicator = Deduplicator()

        # Shared thread pool — one thread per API worker
        self._executor = ThreadPoolExecutor(max_workers=settings.api_workers)
        self._worker = QAWorker(self._qa_model, self._executor)

        logger.info(
            "AuditDispatcher initialized",
            extra={"workers": settings.api_workers},
        )

    def _expand_answer_if_needed(self, qa_result: QAResult, chunk_text: str) -> str:
        """
        Intelligently expand short answers to provide more context.
        Generic approach that works for any document.
        
        Strategies:
        1. If answer is already substantial, keep it
        2. If answer is very short, find the surrounding sentence
        3. If answer is still short, expand to context window around answer
        4. Fallback to first meaningful sentence from chunk
        """
        # Strategy 1: Keep substantial answers (over 100 chars)
        if len(qa_result.answer) >= 100:
            return qa_result.answer
        
        if not chunk_text or len(chunk_text) == 0:
            return qa_result.answer
        
        # Strategy 2: Find the sentence containing the answer
        # Split into sentences (supports ., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        
        for sentence in sentences:
            if qa_result.answer.lower() in sentence.lower():
                cleaned = sentence.strip()
                if len(cleaned) > len(qa_result.answer):
                    logger.debug(
                        "Expanded answer using surrounding sentence",
                        extra={
                            "original": qa_result.answer[:50],
                            "expanded": cleaned[:100],
                        }
                    )
                    return cleaned
        
        # Strategy 3: Get context window around the answer
        answer_pos = chunk_text.lower().find(qa_result.answer.lower())
        if answer_pos != -1:
            # Get window of text (150 chars before, 200 chars after)
            start = max(0, answer_pos - 150)
            end = min(len(chunk_text), answer_pos + 200)
            
            # Expand to full sentence boundaries
            while start > 0 and chunk_text[start] not in '.!?':
                start -= 1
            if start > 0:
                start += 1  # Skip the punctuation
            
            while end < len(chunk_text) and chunk_text[end] not in '.!?':
                end += 1
            if end < len(chunk_text):
                end += 1
            
            expanded = chunk_text[start:end].strip()
            if len(expanded) > len(qa_result.answer):
                logger.debug(
                    "Expanded answer using context window",
                    extra={
                        "original": qa_result.answer[:50],
                        "expanded": expanded[:100],
                    }
                )
                return expanded
        
        # Strategy 4: Return first meaningful sentence from chunk
        for sentence in sentences:
            # Find a sentence that's not too short (more than 40 chars)
            # and not just numbers or single words
            cleaned = sentence.strip()
            if len(cleaned) > 40 and not cleaned[0].isdigit():
                logger.debug(
                    "Using first meaningful sentence as answer",
                    extra={"sentence": cleaned[:100]}
                )
                return cleaned
        
        # Strategy 5: Fallback to full chunk (truncated)
        if len(chunk_text) > 300:
            return chunk_text[:300] + "..."
        
        return qa_result.answer

    async def dispatch(self, query: str) -> DispatchResult:
        """
        Run the full audit pipeline for one query.

        Args:
            query — the audit question string

        Returns:
            DispatchResult with ranked VerifiedAnswers.

        Raises:
            NoAnswerFoundError — if no span passes abstention
                                 even after RAKE fallback.
        """
        logger.info("Dispatch started", extra={"query": query[:80]})

        # ── Pass 1: Standard retrieval + QA ───────────────────────────────────
        chunks = await self._retriever.retrieve(query)
        worker_results = await self._run_workers(query, chunks)
        qa_results = [r.qa_result for r in worker_results if r.success and r.qa_result]
        accepted = self._abstention.filter(qa_results)

        rake_used = False

        # ── Pass 2: RAKE fallback (if all spans rejected) ─────────────────────
        if not accepted:
            logger.info(
                "All spans rejected — triggering RAKE fallback",
                extra={"query": query[:80]},
            )
            reformulated = self._rake.reformulate(query)

            if reformulated != query:
                rake_used = True
                chunks = await self._retriever.retrieve(reformulated)
                worker_results = await self._run_workers(reformulated, chunks)
                qa_results = [
                    r.qa_result for r in worker_results
                    if r.success and r.qa_result
                ]
                accepted = self._abstention.filter(qa_results)

        # ── No answer found ───────────────────────────────────────────────────
        if not accepted:
            raise NoAnswerFoundError(query)

        # ── Deduplicate ───────────────────────────────────────────────────────
        unique = self._deduplicator.deduplicate(accepted)

        # ── Build chunk lookup for source attribution ──────────────────────────
        chunk_map = self._build_chunk_map(worker_results)

        # ── Build VerifiedAnswer list ──────────────────────────────────────────
        from core.utils import normalize_span, sha256_hash

        answers: list[VerifiedAnswer] = []
        for qa_result in unique:
            source_chunk = chunk_map.get(id(qa_result))
            if source_chunk is None:
                continue

            # Expand short answers if needed (generic approach)
            expanded_text = self._expand_answer_if_needed(qa_result, source_chunk.text)

            answers.append(VerifiedAnswer(
                text=expanded_text,
                span_score=round(qa_result.span_score, 6),
                null_score=round(qa_result.null_score, 6),
                filename=source_chunk.filename,
                page_number=source_chunk.page_number,
                bbox=source_chunk.bbox,
                chunk_text=source_chunk.text,
                span_hash=sha256_hash(normalize_span(qa_result.answer)),
                rake_used=rake_used,
            ))

        total_searched = len(worker_results)

        logger.info(
            "Dispatch complete",
            extra={
                "query": query[:80],
                "answers": len(answers),
                "rake_used": rake_used,
                "chunks_searched": total_searched,
            },
        )

        return DispatchResult(
            answers=answers,
            query=query,
            rake_used=rake_used,
            total_chunks_searched=total_searched,
        )

    async def _run_workers(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> list[WorkerResult]:
        """
        Run QA concurrently on all chunks using asyncio.gather.

        All workers start simultaneously. Results preserve
        the order of the input chunks.
        """
        if not chunks:
            return []

        tasks = [self._worker.run(question, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def _build_chunk_map(
        self,
        worker_results: list[WorkerResult],
    ) -> dict[int, RetrievedChunk]:
        """
        Build a map from QAResult object id → RetrievedChunk.
        Used to trace each answer back to its source document.
        """
        mapping = {}
        for wr in worker_results:
            if wr.success and wr.qa_result is not None:
                mapping[id(wr.qa_result)] = wr.chunk
        return mapping

    def shutdown(self) -> None:
        """Cleanly shut down the thread pool."""
        self._executor.shutdown(wait=True)
        logger.info("AuditDispatcher thread pool shut down")