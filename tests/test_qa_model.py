"""
tests/test_qa_model.py
======================
Tests for auditor/qa_model.py

Covers:
  - QAModel loads successfully
  - predict() returns a QAResult
  - Extracted answer exists within the context string
  - span_score and null_score are numeric
  - Empty question raises QAModelError
  - Empty context raises QAModelError
"""

import pytest
from auditor.qa_model import QAModel, QAResult
from core.exceptions import QAModelError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qa_model():
    """Load QA model once for all tests in this module."""
    return QAModel()


ANSWERABLE_CONTEXT = (
    "The product warranty covers all manufacturing defects for a period "
    "of two years from the date of purchase. Customers must retain their "
    "original receipt to make a warranty claim."
)

UNANSWERABLE_CONTEXT = (
    "The sky appears blue because of Rayleigh scattering of sunlight "
    "through the atmosphere."
)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQAModel:

    def test_model_loads(self, qa_model):
        """QAModel should initialize without errors."""
        assert qa_model is not None

    def test_predict_returns_qa_result(self, qa_model):
        """predict() should return a QAResult instance."""
        result = qa_model.predict(
            question="What is the warranty period?",
            context=ANSWERABLE_CONTEXT,
        )
        assert isinstance(result, QAResult)

    def test_answer_within_context(self, qa_model):
        """Extracted answer should be a substring of the context."""
        result = qa_model.predict(
            question="What is the warranty period?",
            context=ANSWERABLE_CONTEXT,
        )
        assert result.answer in ANSWERABLE_CONTEXT

    def test_span_score_is_numeric(self, qa_model):
        """span_score should be a finite float."""
        result = qa_model.predict(
            question="What is the warranty period?",
            context=ANSWERABLE_CONTEXT,
        )
        assert isinstance(result.span_score, float)
        assert not (result.span_score != result.span_score)  # not NaN

    def test_null_score_is_numeric(self, qa_model):
        """null_score should be a finite float."""
        result = qa_model.predict(
            question="What is the warranty period?",
            context=ANSWERABLE_CONTEXT,
        )
        assert isinstance(result.null_score, float)

    def test_char_offsets_valid(self, qa_model):
        """char_start and char_end should be valid offsets into context."""
        result = qa_model.predict(
            question="How long is the warranty?",
            context=ANSWERABLE_CONTEXT,
        )
        assert 0 <= result.char_start < result.char_end
        assert result.char_end <= len(ANSWERABLE_CONTEXT)

    def test_answerable_question_high_score(self, qa_model):
        """A clearly answerable question should produce span_score > null_score."""
        result = qa_model.predict(
            question="What is the warranty period?",
            context=ANSWERABLE_CONTEXT,
        )
        # For a clearly answerable question the span should beat null
        assert result.span_score > result.null_score

    def test_empty_question_raises(self, qa_model):
        """Empty question should raise QAModelError."""
        with pytest.raises(QAModelError):
            qa_model.predict(question="", context=ANSWERABLE_CONTEXT)

    def test_empty_context_raises(self, qa_model):
        """Empty context should raise QAModelError."""
        with pytest.raises(QAModelError):
            qa_model.predict(
                question="What is the warranty?",
                context="",
            )

    def test_has_answer_false_by_default(self, qa_model):
        """QAResult.has_answer should be False before abstention evaluation."""
        result = qa_model.predict(
            question="What is the warranty period?",
            context=ANSWERABLE_CONTEXT,
        )
        assert result.has_answer is False  # set by abstention, not qa_model