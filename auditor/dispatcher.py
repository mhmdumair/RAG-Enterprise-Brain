"""
auditor/dispatcher.py
=====================
Parallel QA dispatcher — Boundary 2 entry point.

Change from previous version:
    CrossEncoderReranker inserted between retrieval and QA.

    Updated flow:
        1. Retrieve Top 20 chunks via ANN search (FAISS)
        2. Cross-Encoder scores each (query, chunk) pair
        3. Re-ranked Top 5 passed to parallel QA workers
        4. Abstention filter (S_span > S_null + τ_ans)
        5. RAKE fallback if all spans rejected
        6. Deduplication
        7. Return VerifiedAnswer list

    If rerank_enabled=False in config, step 2-3 are skipped
    and raw FAISS Top 5 is used instead (fallback/debug mode).
"""

import asyncio
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from motor.motor_asyncio import AsyncIOMotorDatabase

from core.config import settings
from core.logger import get_logger
from core.exceptions import NoAnswerFoundError
from core.utils import normalize_span, sha256_hash
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from auditor.retriever import Retriever, RetrievedChunk
from auditor.reranker import CrossEncoderReranker
from auditor.qa_model import QAModel
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
        text                — the extracted answer string
        span_score          — RoBERTa confidence score
        null_score          — RoBERTa no-answer score
        rerank_score        — Cross-Encoder relevance score for source chunk
        filename            — source PDF filename
        page_number         — source page (1-based)
        bbox                — normalized BBox for frontend highlighting
        chunk_text          — full chunk context
        span_hash           — SHA-256 of normalized answer
        rake_used           — True if RAKE fallback triggered
    """
    text:         str
    span_score:   float
    null_score:   float
    rerank_score: float
    filename:     str
    page_number:  int
    bbox:         dict
    chunk_text:   str
    span_hash:    str
    rake_used:    bool = False


@dataclass
class DispatchResult:
    """
    Full result returned by the dispatcher.

    Fields:
        answers               — ranked list of VerifiedAnswer
        query                 — original query string
        rake_used             — True if RAKE fallback was triggered
        total_chunks_searched — how many chunks were evaluated by QA
        reranked              — True if Cross-Encoder re-ranking was applied
    """
    answers:               list[VerifiedAnswer]
    query:                 str
    rake_used:             bool
    total_chunks_searched: int
    reranked:              bool


# ── Dispatcher ────────────────────────────────────────────────────────────────

class AuditDispatcher:
    """
    Parallel audit dispatcher with Cross-Encoder re-ranking.

    Components are injected and shared across requests.
    The thread pool is shared for efficient CPU utilization.

    Usage:
        dispatcher = AuditDispatcher(embedder, faiss_index, db)
        result = await dispatcher.dispatch("What is the warranty period?")
    """

    def __init__(
        self,
        embedder:    Embedder,
        faiss_index: FAISSIndex,
        db:          AsyncIOMotorDatabase,
    ):
        self._retriever   = Retriever(embedder, faiss_index, db)
        self._qa_model    = QAModel()
        self._abstention  = AbstentionFilter()
        self._rake        = RAKEFallback()
        self._deduplicator = Deduplicator()

        # Shared thread pool — CPU-bound inference runs here
        self._executor = ThreadPoolExecutor(max_workers=settings.api_workers)
        self._worker   = QAWorker(self._qa_model, self._executor)

        # Cross-Encoder re-ranker — loaded only if enabled
        self._reranker: CrossEncoderReranker | None = None
        if settings.rerank_enabled:
            self._reranker = CrossEncoderReranker()

        logger.info(
            "AuditDispatcher initialized",
            extra={
                "workers":         settings.api_workers,
                "rerank_enabled":  settings.rerank_enabled,
                "retrieval_top_k": settings.retrieval_top_k,
                "qa_top_k":        settings.top_k_chunks,
            },
        )

    # ── Public method ─────────────────────────────────────────────────────────

    async def dispatch(self, query: str) -> DispatchResult:
        """
        Run the full audit pipeline for one query.

        Flow:
            1. Retrieve Top 20 chunks (FAISS ANN search)
            2. Re-rank with Cross-Encoder → Top 5 (if enabled)
            3. Run QA workers concurrently on Top 5
            4. Apply abstention filter
            5. RAKE fallback if all rejected
            6. Deduplicate
            7. Return VerifiedAnswer list

        Args:
            query — the audit question string

        Returns:
            DispatchResult with ranked VerifiedAnswers.

        Raises:
            NoAnswerFoundError — if no span passes abstention
                                 even after RAKE fallback.
        """
        logger.info(
            "Dispatch started",
            extra={"query": query[:80]},
        )

        # ── Pass 1: Retrieve → Re-rank → QA ──────────────────────────────────
        chunks, reranked = await self._retrieve_and_rerank(query)
        worker_results   = await self._run_workers(query, chunks)
        qa_results       = [
            r.qa_result for r in worker_results
            if r.success and r.qa_result
        ]
        accepted = self._abstention.filter(qa_results)

        rake_used = False

        # ── Pass 2: RAKE fallback ─────────────────────────────────────────────
        if not accepted:
            logger.info(
                "All spans rejected — triggering RAKE fallback",
                extra={"query": query[:80]},
            )
            reformulated = self._rake.reformulate(query)

            if reformulated != query:
                rake_used = True
                chunks, reranked = await self._retrieve_and_rerank(reformulated)
                worker_results   = await self._run_workers(reformulated, chunks)
                qa_results       = [
                    r.qa_result for r in worker_results
                    if r.success and r.qa_result
                ]
                accepted = self._abstention.filter(qa_results)

        # ── No answer found ───────────────────────────────────────────────────
        if not accepted:
            raise NoAnswerFoundError(query)

        # ── Deduplicate ───────────────────────────────────────────────────────
        unique = self._deduplicator.deduplicate(accepted)

        # ── Build chunk lookup for source attribution ─────────────────────────
        chunk_map = self._build_chunk_map(worker_results)

        # ── Build VerifiedAnswer list ─────────────────────────────────────────
        answers: list[VerifiedAnswer] = []
        for qa_result in unique:
            source_chunk = chunk_map.get(id(qa_result))
            if source_chunk is None:
                continue

            # EXPAND THE ANSWER HERE
            expanded_text = self._expand_answer_if_needed(qa_result, source_chunk.text)

            answers.append(VerifiedAnswer(
                text=expanded_text,
                span_score=round(qa_result.span_score, 6),
                null_score=round(qa_result.null_score, 6),
                rerank_score=round(getattr(source_chunk, "rerank_score", 0.0), 6),
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
                "query":           query[:80],
                "answers":         len(answers),
                "rake_used":       rake_used,
                "reranked":        reranked,
                "chunks_searched": total_searched,
            },
        )

        return DispatchResult(
            answers=answers,
            query=query,
            rake_used=rake_used,
            total_chunks_searched=total_searched,
            reranked=reranked,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _expand_answer_if_needed(self, qa_result, chunk_text: str) -> str:
        """
        Intelligently expand short answers to provide more context.
        Generic approach that works for any document.
        
        Strategies:
        1. If answer is already substantial, keep it
        2. If answer is very short, find the surrounding sentence
        3. If answer is still short, expand to context window around answer
        4. Fallback to first meaningful sentence from chunk
        """
        import re
        
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

    async def _retrieve_and_rerank(
        self,
        query: str,
    ) -> tuple[list[RetrievedChunk], bool]:
        """
        Retrieve candidate chunks then apply Cross-Encoder re-ranking.

        Steps:
            1. FAISS retrieves Top 20 (retrieval_top_k)
            2. Cross-Encoder scores each (query, chunk) pair
            3. Returns Top 5 (top_k_chunks) sorted by rerank_score

        If rerank_enabled=False, returns raw FAISS Top 5 directly.

        Args:
            query — the question string

        Returns:
            (chunks, reranked_flag)
            chunks       — list[RetrievedChunk] ready for QA workers
            reranked_flag— True if Cross-Encoder was applied
        """
        # Step 1: FAISS retrieval — wide candidate pool
        chunks = await self._retriever.retrieve(query)

        if not chunks:
            return [], False

        # Step 2: Cross-Encoder re-ranking (if enabled)
        if self._reranker is not None and settings.rerank_enabled:
            loop = asyncio.get_event_loop()
            # Run CPU-bound Cross-Encoder in thread pool
            chunks = await loop.run_in_executor(
                self._executor,
                self._reranker.rerank,
                query,
                chunks,
                settings.top_k_chunks,
            )
            return chunks, True

        # Fallback: no re-ranker — slice raw FAISS results to top_k_chunks
        return chunks[:settings.top_k_chunks], False

    async def _run_workers(
        self,
        question: str,
        chunks:   list[RetrievedChunk],
    ) -> list[WorkerResult]:
        """
        Run QA concurrently on all chunks using asyncio.gather.
        """
        if not chunks:
            return []

        tasks   = [self._worker.run(question, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def _build_chunk_map(
        self,
        worker_results: list[WorkerResult],
    ) -> dict[int, RetrievedChunk]:
        """
        Map QAResult object id → RetrievedChunk for source attribution.
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