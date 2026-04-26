"""
api/routes/debug.py
===================
Debug endpoints for troubleshooting the RAG pipeline.

Exposes internal state of:
  - Retriever (what chunks are found)
  - QA Model (what scores are produced)
  - Abstention Filter (why answers are rejected)
"""

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from core.logger import get_logger
from brain.embedder import Embedder
from brain.indexer import FAISSIndex
from auditor.retriever import Retriever
from auditor.qa_model import QAModel
from auditor.abstention import AbstentionFilter
from api.dependencies import get_embedder, get_faiss_index, get_db
from api.schemas import QueryRequest

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/debug/query-pipeline",
    summary="Debug query pipeline - see retrieval and QA scores",
    tags=["Debug"],
)
async def debug_query_pipeline(
    request: QueryRequest,
    embedder: Embedder = Depends(get_embedder),
    faiss_index: FAISSIndex = Depends(get_faiss_index),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Step through the query pipeline and return internal state.
    
    Shows:
    - Retrieved chunks (text, distance, filename, page)
    - QA model outputs (answer, span_score, null_score)
    - Abstention filter results (why each answer was accepted/rejected)
    """
    query = request.query
    
    # ── Step 1: Retrieve chunks ──────────────────────────────────────────
    retriever = Retriever(embedder, faiss_index, db)
    chunks = await retriever.retrieve(query, k=5)
    
    if not chunks:
        return {
            "query": query,
            "status": "no_chunks",
            "message": "Retriever found 0 chunks. Check FAISS index or query.",
            "chunks": []
        }
    
    logger.info(f"Retrieved {len(chunks)} chunks for debugging")
    
    # ── Step 2: Run QA on each chunk ─────────────────────────────────────
    qa_model = QAModel()
    qa_results = []
    
    for i, chunk in enumerate(chunks):
        try:
            qa_result = qa_model.predict(query, chunk.text)
            qa_results.append({
                "chunk_index": i,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "distance": round(chunk.distance, 4),
                "chunk_text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "answer": qa_result.answer,
                "span_score": round(qa_result.span_score, 6),
                "null_score": round(qa_result.null_score, 6),
                "score_margin": round(qa_result.span_score - qa_result.null_score, 6),
                "start": qa_result.start,
                "end": qa_result.end,
            })
        except Exception as e:
            qa_results.append({
                "chunk_index": i,
                "filename": chunk.filename,
                "error": str(e)
            })
    
    # ── Step 3: Apply abstention filter ──────────────────────────────────
    abstention = AbstentionFilter()
    
    # Build minimal QAResult objects for the records
    from auditor.qa_model import QAResult
    qa_result_objs = []
    for i, chunk in enumerate(chunks):
        try:
            qa_result = qa_model.predict(query, chunk.text)
            qa_result_objs.append(qa_result)
        except Exception:
            pass
    
    accepted = abstention.filter(qa_result_objs)
    
    # ── Step 4: Build response ───────────────────────────────────────────
    return {
        "query": query,
        "status": "ok",
        "tau_ans": abstention.tau,
        "total_chunks_retrieved": len(chunks),
        "total_qa_results": len(qa_results),
        "total_accepted": len(accepted),
        "qa_results": qa_results,
        "accepted_answers": [
            {
                "text": r.answer,
                "span_score": round(r.span_score, 6),
                "null_score": round(r.null_score, 6),
                "margin": round(r.span_score - r.null_score, 6),
            }
            for r in accepted
        ],
        "recommendation": 
            "✓ Answers found - query should work" if accepted else 
            "✗ No answers passed threshold" if qa_results else
            "✗ No chunks retrieved - try different query"
    }
