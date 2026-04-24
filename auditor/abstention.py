"""
auditor/abstention.py
=====================
Span acceptance/rejection logic.

The abstention condition:
    Accept span  if  S_span > S_null + τ_ans
    Reject span  if  S_span ≤ S_null + τ_ans

Where:
    S_span  = best start_logit + end_logit score for the extracted span
    S_null  = start_logit[CLS] + end_logit[CLS] (the "no answer" score)
    τ_ans   = calibrated confidence margin (default 0.1 from config)

Why this works:
    RoBERTa was fine-tuned on SQuAD2.0 which contains unanswerable
    questions. The CLS token score represents the model's confidence
    that NO answer exists. If the span score doesn't clearly beat
    the null score by at least τ_ans, we abstain rather than guess.

This is the mathematical elimination of hallucination.
"""

from core.config import settings
from core.logger import get_logger
from auditor.qa_model import QAResult

logger = get_logger(__name__)


class AbstentionFilter:
    """
    Applies the S_span > S_null + τ_ans threshold to QAResults.

    Usage:
        filt = AbstentionFilter()
        result = filt.evaluate(qa_result)   # stamps has_answer on result
        accepted = filt.filter(results)     # returns only accepted results
    """

    def __init__(self, tau: float | None = None):
        self._tau = tau if tau is not None else settings.tau_ans
        logger.info(
            "AbstentionFilter initialized",
            extra={"tau_ans": self._tau},
        )

    def evaluate(self, result: QAResult) -> QAResult:
        """
        Stamp has_answer on a QAResult based on the abstention condition.

        Modifies result in place and returns it.

        The condition:
            has_answer = (S_span > S_null + τ_ans) AND answer is non-empty

        Args:
            result — QAResult from QAModel.predict()

        Returns:
            The same QAResult with has_answer set.
        """
        passes_threshold = result.span_score > (result.null_score + self._tau)
        has_content = bool(result.answer and result.answer.strip())

        result.has_answer = passes_threshold and has_content

        logger.debug(
            "Abstention evaluated",
            extra={
                "answer": result.answer[:50] if result.answer else "",
                "span_score": round(result.span_score, 4),
                "null_score": round(result.null_score, 4),
                "threshold": round(result.null_score + self._tau, 4),
                "accepted": result.has_answer,
            },
        )

        return result

    def filter(self, results: list[QAResult]) -> list[QAResult]:
        """
        Evaluate a list of QAResults and return only accepted ones.

        Args:
            results — list of QAResult from multiple chunks

        Returns:
            Filtered list where has_answer is True, sorted by
            span_score descending (most confident first).
        """
        evaluated = [self.evaluate(r) for r in results]
        accepted = [r for r in evaluated if r.has_answer]
        accepted.sort(key=lambda r: r.span_score, reverse=True)

        logger.info(
            "Abstention filter applied",
            extra={
                "total": len(results),
                "accepted": len(accepted),
                "rejected": len(results) - len(accepted),
            },
        )

        return accepted

    @property
    def tau(self) -> float:
        return self._tau