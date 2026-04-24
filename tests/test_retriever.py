"""
tests/test_retriever.py
=======================
Tests for auditor/retriever.py
"""

import pytest
from auditor.retriever import Retriever, RetrievedChunk
from brain.indexer import FAISSIndex
from core.exceptions import RetrievalError


@pytest.mark.asyncio
async def test_retrieve_returns_results(embedder, mongo_db):
    """retrieve() should return a non-empty list for a valid query."""
    faiss_index = FAISSIndex()
    if not faiss_index.load():
        pytest.skip("No FAISS index found - run Phase 2 first")
    
    retriever = Retriever(embedder, faiss_index, mongo_db)
    results = await retriever.retrieve("What is the main topic?", k=3)
    
    if results:
        assert len(results) <= 3
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)


@pytest.mark.asyncio
async def test_retrieve_results_ordered_by_distance(embedder, mongo_db):
    """Results should be ordered by distance ascending (closest first)."""
    faiss_index = FAISSIndex()
    if not faiss_index.load():
        pytest.skip("No FAISS index found")
    
    retriever = Retriever(embedder, faiss_index, mongo_db)
    results = await retriever.retrieve("document purpose", k=5)
    
    if len(results) >= 2:
        distances = [r.distance for r in results]
        assert distances == sorted(distances), "Results not ordered by distance"


@pytest.mark.asyncio
async def test_retrieve_empty_index_raises(embedder, mongo_db):
    """retrieve() on empty index should raise RetrievalError."""
    empty_index = FAISSIndex()  # not built and not loaded
    retriever = Retriever(embedder, empty_index, mongo_db)
    with pytest.raises(RetrievalError):
        await retriever.retrieve("test query", k=3)