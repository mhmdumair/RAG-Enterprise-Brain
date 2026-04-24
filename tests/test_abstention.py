"""
tests/test_abstention.py
========================
Tests for auditor/abstention.py, auditor/rake_fallback.py,
and auditor/deduplicator.py

Covers:
  - AbstentionFilter accepts span when S_span > S_null + tau
  - AbstentionFilter rejects span when S_span <= S_null + tau
  - AbstentionFilter.filter() returns only accepted results
  - AbstentionFilter.filter() sorts by span_score descending
  - RAKEFallback extracts keywords from a query
  - RAKEFallback.reformulate() returns shorter keyword string
  - Deduplicator removes exact duplicate spans
  - Deduplicator keeps highest-scoring version of duplicate
"""

import pytest
from auditor.qa_model import QAResult
from auditor.abstention import AbstentionFilter
from auditor.rake_fallback import RAKEFallback
from auditor.deduplicator import Deduplicator


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_result(answer: str, span_score: float, null_score: float) -> QAResult:
    return QAResult(
        answer=answer,
        span_score=span_score,
        null_score=null_score,
        char_start=0,
        char_end=len(answer),
    )


# ── Abstention tests ──────────────────────────────────────────────────────────

class TestAbstentionFilter:

    def setup_method(self):
        self.filt = AbstentionFilter(tau=0.1)

    def test_accepts_when_span_clearly_beats_null(self):
        """Span with score well above null + tau should be accepted."""
        result = make_result("two years", span_score=2.0, null_score=0.5)
        evaluated = self.filt.evaluate(result)
        assert evaluated.has_answer is True

    def test_rejects_when_span_below_threshold(self):
        """Span with score below null + tau should be rejected."""
        result = make_result("two years", span_score=0.5, null_score=0.5)
        evaluated = self.filt.evaluate(result)
        assert evaluated.has_answer is False

    def test_rejects_empty_answer(self):
        """Empty answer string should be rejected regardless of score."""
        result = make_result("", span_score=5.0, null_score=0.1)
        evaluated = self.filt.evaluate(result)
        assert evaluated.has_answer is False

    def test_rejects_whitespace_answer(self):
        """Whitespace-only answer should be rejected."""
        result = make_result("   ", span_score=5.0, null_score=0.1)
        evaluated = self.filt.evaluate(result)
        assert evaluated.has_answer is False

    def test_filter_returns_only_accepted(self):
        """filter() should return only results where has_answer is True."""
        results = [
            make_result("good answer", span_score=3.0, null_score=0.5),
            make_result("bad answer", span_score=0.4, null_score=0.5),
            make_result("another good", span_score=2.0, null_score=0.5),
        ]
        accepted = self.filt.filter(results)
        assert len(accepted) == 2
        assert all(r.has_answer for r in accepted)

    def test_filter_sorts_by_score_descending(self):
        """filter() should return results sorted by span_score descending."""
        results = [
            make_result("low score", span_score=1.5, null_score=0.5),
            make_result("high score", span_score=3.0, null_score=0.5),
            make_result("mid score", span_score=2.0, null_score=0.5),
        ]
        accepted = self.filt.filter(results)
        scores = [r.span_score for r in accepted]
        assert scores == sorted(scores, reverse=True)

    def test_filter_empty_input_returns_empty(self):
        """filter() on empty list should return empty list."""
        assert self.filt.filter([]) == []

    def test_custom_tau_is_respected(self):
        """Higher tau should be stricter — reject spans that lower tau accepts."""
        strict = AbstentionFilter(tau=1.0)
        lenient = AbstentionFilter(tau=0.01)
        result = make_result("borderline", span_score=1.2, null_score=1.0)
        # tau=1.0: 1.2 > 1.0 + 1.0 = 2.0 → False
        # tau=0.01: 1.2 > 1.0 + 0.01 = 1.01 → True
        assert strict.evaluate(result).has_answer is False
        result2 = make_result("borderline", span_score=1.2, null_score=1.0)
        assert lenient.evaluate(result2).has_answer is True


# ── RAKE tests ────────────────────────────────────────────────────────────────

class TestRAKEFallback:

    def setup_method(self):
        self.rake = RAKEFallback()

    def test_extract_keywords_returns_list(self):
        """extract_keywords() should return a list."""
        keywords = self.rake.extract_keywords(
            "What is the maximum torque specification for the engine?"
        )
        assert isinstance(keywords, list)

    def test_reformulate_returns_string(self):
        """reformulate() should return a non-empty string."""
        result = self.rake.reformulate("What is the warranty period for the product?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_reformulated_query_is_shorter_or_equal(self):
        """Reformulated query should typically be shorter than the original."""
        original = "What is the maximum allowable torque specification for bolt fastening?"
        reformulated = self.rake.reformulate(original)
        # Keywords should be fewer words than the full question
        assert len(reformulated.split()) <= len(original.split())

    def test_empty_string_returns_empty(self):
        """Empty input should return empty keywords list."""
        keywords = self.rake.extract_keywords("")
        assert keywords == []


# ── Deduplicator tests ────────────────────────────────────────────────────────

class TestDeduplicator:

    def setup_method(self):
        self.dedup = Deduplicator()

    def test_dedup_removes_exact_duplicates(self):
        """Exact same answer text should be deduplicated to one result."""
        results = [
            make_result("two years", span_score=2.0, null_score=0.5),
            make_result("two years", span_score=1.5, null_score=0.5),
        ]
        unique = self.dedup.deduplicate(results)
        assert len(unique) == 1

    def test_dedup_keeps_highest_score(self):
        """When deduplicating, keep the result with the highest span_score."""
        results = [
            make_result("two years", span_score=1.5, null_score=0.5),
            make_result("two years", span_score=3.0, null_score=0.5),
        ]
        unique = self.dedup.deduplicate(results)
        assert unique[0].span_score == 3.0

    def test_dedup_normalizes_before_hashing(self):
        """'Two years.' and 'two years' should be treated as duplicates."""
        results = [
            make_result("Two years.", span_score=2.0, null_score=0.5),
            make_result("two years", span_score=1.5, null_score=0.5),
        ]
        unique = self.dedup.deduplicate(results)
        assert len(unique) == 1

    def test_dedup_keeps_distinct_answers(self):
        """Different answers should all be kept."""
        results = [
            make_result("two years", span_score=2.0, null_score=0.5),
            make_result("three months", span_score=1.5, null_score=0.5),
            make_result("one decade", span_score=1.0, null_score=0.5),
        ]
        unique = self.dedup.deduplicate(results)
        assert len(unique) == 3

    def test_dedup_sorts_by_score_descending(self):
        """Output should be sorted by span_score descending."""
        results = [
            make_result("answer a", span_score=1.0, null_score=0.1),
            make_result("answer b", span_score=3.0, null_score=0.1),
            make_result("answer c", span_score=2.0, null_score=0.1),
        ]
        unique = self.dedup.deduplicate(results)
        scores = [r.span_score for r in unique]
        assert scores == sorted(scores, reverse=True)

    def test_dedup_empty_input(self):
        """Empty input should return empty list."""
        assert self.dedup.deduplicate([]) == []

    def test_dedup_single_item(self):
        """Single item should be returned as-is."""
        results = [make_result("only answer", span_score=2.0, null_score=0.5)]
        unique = self.dedup.deduplicate(results)
        assert len(unique) == 1