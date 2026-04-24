"""
auditor/rake_fallback.py
========================
RAKE keyword extraction fallback.

Triggered when: all top-K chunks fail the abstention threshold.

Strategy:
    1. Extract the top N keyword phrases from the original query
       using RAKE (score = word frequency × word degree)
    2. Join the top phrases into a reformulated query string
    3. Return the reformulated query for one retry of the retrieval loop

Why RAKE?
    - Pure statistical — no model required
    - Fast (microseconds)
    - Works well for domain-specific noun phrases
    - No external API calls

Example:
    Original:  "What is the maximum allowable torque for bolt fastening?"
    RAKE top3: ["maximum allowable torque", "bolt fastening"]
    Reformulated: "maximum allowable torque bolt fastening"
"""

from rake_nltk import Rake
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class RAKEFallback:
    """
    Reformulates a failed query using RAKE keyword extraction.

    Usage:
        fallback = RAKEFallback()
        new_query = fallback.reformulate("What is the warranty period?")
        # → "warranty period"
    """

    def __init__(self):
        self._rake = Rake(
            max_length=settings.rake_max_words,
            min_length=1,
        )
        self._top_n = settings.rake_top_n
        logger.info(
            "RAKEFallback initialized",
            extra={
                "max_words": settings.rake_max_words,
                "top_n": self._top_n,
            },
        )

    def extract_keywords(self, text: str) -> list[str]:
        """
        Extract top-N keyword phrases from text.

        Args:
            text — the original query string

        Returns:
            List of keyword phrases ordered by RAKE score descending.
            Returns original text split as fallback if RAKE finds nothing.
        """
        if not text.strip():
            return []

        try:
            self._rake.extract_keywords_from_text(text)
            phrases = self._rake.get_ranked_phrases()
            top = phrases[:self._top_n] if phrases else []

            logger.info(
                "RAKE keywords extracted",
                extra={
                    "original": text[:80],
                    "keywords": top,
                },
            )

            return top

        except Exception as exc:
            logger.warning(
                "RAKE extraction failed, using raw query",
                extra={"error": str(exc)},
            )
            return [text]

    def reformulate(self, query: str) -> str:
        """
        Reformulate a query into a keyword-only string for retry.

        Args:
            query — the original failed query

        Returns:
            Keyword-only query string.
            Falls back to the original query if RAKE finds nothing.

        Example:
            "What is the maximum allowable torque for bolt fastening?"
            → "maximum allowable torque bolt fastening"
        """
        keywords = self.extract_keywords(query)

        if not keywords:
            logger.warning(
                "RAKE found no keywords, using original query",
                extra={"query": query},
            )
            return query

        reformulated = " ".join(keywords)

        logger.info(
            "Query reformulated",
            extra={
                "original": query[:80],
                "reformulated": reformulated[:80],
            },
        )

        return reformulated