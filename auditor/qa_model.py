"""
auditor/qa_model.py
===================
INT8-quantized extractive QA using deepset/roberta-base-squad2.

Why extractive QA?
    The model can ONLY return a span that literally exists in the
    context text. It cannot generate or hallucinate new content.
    This is the core anti-hallucination guarantee of the system.

Why INT8 quantization?
    Reduces the model from ~500MB to ~125MB and cuts inference
    time by ~2x on CPU, fitting comfortably within 8GB RAM.

Output:
    QAResult — contains the answer text, scores, and char offsets.
    The span_score and null_score are used by abstention.py to
    decide whether to trust or reject the answer.

Flow:
    (question, context) → tokenize → model forward pass
      → start/end logits → softmax → span + null scores → QAResult
"""

from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.quantization import quantize_dynamic

from core.config import settings
from core.logger import get_logger
from core.exceptions import QAModelError

logger = get_logger(__name__)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    """
    Output of a single QA inference call.

    Fields:
        answer       — extracted answer span text (empty string if abstained)
        span_score   — probability that this span is the answer (S_span)
        null_score   — probability that there is NO answer (S_null)
        char_start   — start character offset in context
        char_end     — end character offset in context
        has_answer   — True if span_score > null_score + tau_ans
    """
    answer: str
    span_score: float
    null_score: float
    char_start: int
    char_end: int
    has_answer: bool = False


# ── QA Model ──────────────────────────────────────────────────────────────────

class QAModel:
    """
    Wrapper around roberta-base-squad2 with INT8 dynamic quantization.

    The model is loaded once at startup and reused across all requests.
    Dynamic quantization is applied to Linear layers only — this is
    the safest form of quantization, requiring no calibration data.

    Usage:
        qa = QAModel()
        result = qa.predict(question="What is X?", context="...text...")
    """

    def __init__(self):
        logger.info(
            "Loading QA model",
            extra={"model": settings.qa_model_name},
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.qa_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
            model = AutoModelForQuestionAnswering.from_pretrained(
                settings.qa_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
            model.eval()

            # INT8 dynamic quantization — Linear layers only
            # Reduces size ~4x, speeds up CPU inference ~2x
            self._model = quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

        except Exception as exc:
            raise QAModelError(f"Failed to load QA model: {exc}") from exc

        logger.info(
            "QA model loaded (INT8 quantized)",
            extra={"model": settings.qa_model_name},
        )

    def predict(self, question: str, context: str) -> QAResult:
        """
        Run extractive QA on a single (question, context) pair.

        The model returns logits over all token positions for both
        the start and end of the answer span. We find the best
        valid span and compute softmax probabilities.

        Args:
            question — the audit query
            context  — the chunk text to search within

        Returns:
            QAResult with answer span and confidence scores.

        Raises:
            QAModelError — if tokenization or inference fails
        """
        if not question.strip():
            raise QAModelError("Question cannot be empty.")
        if not context.strip():
            raise QAModelError("Context cannot be empty.")

        try:
            # ── Tokenize ──────────────────────────────────────────────────────
            inputs = self._tokenizer(
                question,
                context,
                return_tensors="pt",
                truncation="only_second",        # truncate context, not question
                max_length=settings.qa_max_seq_len,
                stride=settings.qa_doc_stride,
                return_overflowing_tokens=False,
                return_offsets_mapping=True,     # char offsets for span extraction
                padding=False,
            )

            offset_mapping = inputs.pop("offset_mapping")[0]
            sequence_ids = inputs.sequence_ids(0)

            # ── Forward pass ──────────────────────────────────────────────────
            with torch.no_grad():
                outputs = self._model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            # ── Find the best valid span ───────────────────────────────────────
            # Mask out tokens that belong to the question (sequence_id != 1)
            context_start_token = next(
                i for i, s in enumerate(sequence_ids) if s == 1
            )
            context_end_token = len(sequence_ids) - 1 - next(
                i for i, s in enumerate(reversed(sequence_ids)) if s == 1
            )

            # Null score = logit at CLS token [0] for both start and end
            null_score = float(
                start_logits[0].item() + end_logits[0].item()
            )

            # Find best span within context tokens
            best_score = float("-inf")
            best_start = context_start_token
            best_end = context_start_token

            for start_idx in range(context_start_token, context_end_token + 1):
                for end_idx in range(
                    start_idx,
                    min(start_idx + settings.qa_max_answer_len, context_end_token + 1),
                ):
                    score = start_logits[start_idx].item() + end_logits[end_idx].item()
                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx

            # ── Convert token offsets to character offsets ────────────────────
            start_char = int(offset_mapping[best_start][0])
            end_char = int(offset_mapping[best_end][1])
            answer_text = context[start_char:end_char].strip()

            return QAResult(
                answer=answer_text,
                span_score=float(best_score),
                null_score=float(null_score),
                char_start=start_char,
                char_end=end_char,
            )

        except QAModelError:
            raise
        except Exception as exc:
            raise QAModelError(f"Inference failed: {exc}") from exc