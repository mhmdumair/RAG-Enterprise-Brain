"""
tests/conftest.py
=================
Pytest configuration and shared fixtures.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Generator, AsyncGenerator

from db.client import get_client, get_database, close_client, ping_database
from brain.embedder import Embedder
from brain.indexer import FAISSIndex

# Ensure we have a PDF for testing
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def mongo_db() -> AsyncGenerator:
    """Create a MongoDB connection for testing."""
    # Ensure client is initialized
    get_client()
    db = get_database()
    yield db
    await close_client()


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Load embedder once for all tests."""
    return Embedder()


@pytest.fixture(scope="session")
def faiss_index():
    """Load FAISS index once for all tests."""
    index = FAISSIndex()
    index.load()  # Try to load existing index
    return index


@pytest.fixture(scope="session")
def test_pdf_path() -> Path:
    """Return path to a test PDF."""
    pdfs = list(FIXTURES_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF files in tests/fixtures/")
    return pdfs[0]


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing."""
    return "This is a sample text for testing the QA model. " \
           "The warranty period is two years from purchase date."


@pytest.fixture
def sample_question() -> str:
    """Return sample question for testing."""
    return "What is the warranty period?"


@pytest.fixture
def sample_answer() -> str:
    """Return expected answer for sample question."""
    return "two years"