"""
auditor/deduplicator.py
=======================
SHA-256 span deduplication.

Problem:
    When multiple overlapping chunks contain the same answer text,
    the QA model extracts the same span multiple times with slightly
    different scores. The user should see each unique answer once.

Solution:
    Normalize each answer span → compute SHA-256 hash → keep only
    the highest-scoring result per unique hash.

Normalization:
    lowercase + strip punctuation + collapse whitespace
    So "Two years." and "two years" → same hash → deduplicated.
"""

from core.logger import get_logger
from core.utils import normalize_span, sha256_hash
from auditor.qa_model import QAResult

logger = get_logger(__name__)


class Deduplicator:
    """
    Removes duplicate answer spans from a list of QAResults.

    Usage:
        dedup = Deduplicator()
        unique = dedup.deduplicate(accepted_results)
    """

    def deduplicate(self, results: list[QAResult]) -> list[QAResult]:
        """
        Deduplicate results by normalized answer text.

        For duplicate spans, keep the one with the highest span_score.
        Returns results sorted by span_score descending.

        Args:
            results — list of accepted QAResult objects (has_answer=True)

        Returns:
            Deduplicated list, highest confidence first.
        """
        if not results:
            return []

        seen: dict[str, QAResult] = {}

        for result in results:
            normalized = normalize_span(result.answer)
            span_hash = sha256_hash(normalized)

            if span_hash not in seen:
                seen[span_hash] = result
            else:
                # Keep the higher-scoring version
                if result.span_score > seen[span_hash].span_score:
                    seen[span_hash] = result

        unique = sorted(seen.values(), key=lambda r: r.span_score, reverse=True)

        logger.info(
            "Deduplication complete",
            extra={
                "input_count": len(results),
                "unique_count": len(unique),
                "removed_count": len(results) - len(unique),
            },
        )

        return unique