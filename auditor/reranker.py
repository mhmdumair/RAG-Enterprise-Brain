"""
auditor/reranker.py
===================
Cross-Encoder Re-Ranker — the "Judge" stage.

Role in the pipeline:
    FAISS (Bi-Encoder) retrieves a wide candidate pool fast (Top 20).
    The Cross-Encoder scores each (query, chunk) pair together using
    full self-attention — far more accurate than vector similarity alone.
    The Top 5 re-ranked chunks are passed to RoBERTa QA.

Why Cross-Encoder is more accurate than Bi-Encoder for re-ranking:
    Bi-Encoder encodes query and chunk SEPARATELY into vectors.
    Cross-Encoder encodes query and chunk TOGETHER — every token in
    the query attends to every token in the chunk. This lets it detect
    nuanced relevance that vector similarity misses.

    Example:
        Query: "What is the maximum torque for M8 bolts?"
        Bi-Encoder may rank "bolt fastening procedures" highly (similar topic).
        Cross-Encoder correctly ranks "M8 bolt torque spec: 25 Nm" highest
        because it sees the query and chunk tokens interact.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Size: ~66MB (fits easily within 8GB RAM)
    - Trained on MS MARCO — real search query / passage relevance
    - Inference: ~100ms for 20 pairs on CPU

Flow:
    list[RetrievedChunk] + query → score each pair → sort → Top K
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.config import settings
from core.logger import get_logger
from core.exceptions import AuditorError
from auditor.retriever import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks a list of RetrievedChunks using a Cross-Encoder model.

    The model is loaded once at startup and reused across all requests.
    Inference runs synchronously — call from a thread pool in async contexts.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self):
        logger.info(
            "Loading Cross-Encoder re-ranker",
            extra={"model": settings.rerank_model_name},
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.rerank_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.rerank_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
            self._model.eval()

        except Exception as exc:
            raise AuditorError(
                f"Failed to load Cross-Encoder model "
                f"'{settings.rerank_model_name}': {exc}"
            ) from exc

        logger.info(
            "Cross-Encoder re-ranker loaded",
            extra={"model": settings.rerank_model_name},
        )

    def rerank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_k:  int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Score each (query, chunk) pair and return the top_k highest-scoring chunks.

        Each chunk is scored independently — the Cross-Encoder sees the full
        query and the full chunk text together in one forward pass.

        The rerank_score is stored on each RetrievedChunk so downstream
        components can use it for display or further filtering.

        Args:
            query  — the audit question string
            chunks — candidate chunks from FAISS retrieval (Top 20)
            top_k  — number of chunks to return after re-ranking
                     defaults to settings.top_k_chunks (5)

        Returns:
            List of RetrievedChunk objects sorted by rerank_score descending,
            truncated to top_k. Original FAISS distances are preserved.

        Raises:
            AuditorError — if model inference fails
        """
        if not chunks:
            return []

        top_k = top_k or settings.top_k_chunks

        # Can't return more than we have
        top_k = min(top_k, len(chunks))

        logger.info(
            "Re-ranking chunks",
            extra={
                "query":      query[:80],
                "candidates": len(chunks),
                "top_k":      top_k,
            },
        )

        # ── Score all (query, chunk) pairs ────────────────────────────────────
        scores = self._score_pairs(query, chunks)

        # ── Attach scores and sort descending ─────────────────────────────────
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # ── Build result — attach rerank_score to each chunk ──────────────────
        reranked: list[RetrievedChunk] = []
        for chunk, score in scored_chunks[:top_k]:
            # Attach score directly on the chunk object
            # RetrievedChunk is a dataclass — we set the attribute directly
            chunk.rerank_score = score
            reranked.append(chunk)

        logger.info(
            "Re-ranking complete",
            extra={
                "top_score":    round(float(reranked[0].rerank_score), 4) if reranked else 0,
                "bottom_score": round(float(reranked[-1].rerank_score), 4) if reranked else 0,
                "returned":     len(reranked),
            },
        )

        return reranked

    def _score_pairs(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
    ) -> list[float]:
        """
        Run Cross-Encoder inference on all (query, chunk) pairs.

        Batches all pairs into a single tokenizer call for efficiency.
        The model outputs a single relevance logit per pair — higher is better.

        Args:
            query  — audit question string
            chunks — candidate chunks to score

        Returns:
            List of float scores, one per chunk, in the same order as input.

        Raises:
            AuditorError — if tokenization or inference fails
        """
        try:
            # Build list of (query, chunk_text) pairs
            pairs = [(query, chunk.text) for chunk in chunks]

            # Tokenize all pairs in one batch
            inputs = self._tokenizer(
                [p[0] for p in pairs],   # queries
                [p[1] for p in pairs],   # chunk texts
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Forward pass — no gradients needed at inference
            with torch.no_grad():
                outputs = self._model(**inputs)

            # outputs.logits shape: (num_pairs, 1) or (num_pairs, num_labels)
            # For ms-marco models: single logit per pair, higher = more relevant
            logits = outputs.logits

            if logits.shape[-1] == 1:
                # Single logit output — squeeze to 1D
                scores = logits.squeeze(-1).tolist()
            else:
                # Multi-label output — take the positive class score (index 1)
                scores = logits[:, 1].tolist()

            # Ensure list of floats
            if isinstance(scores, float):
                scores = [scores]

            return [float(s) for s in scores]

        except Exception as exc:
            raise AuditorError(
                f"Cross-Encoder inference failed: {exc}"
            ) from exc