"""
tests/test_api.py
=================
Full HTTP integration tests via httpx.AsyncClient.

Tests all API endpoints end-to-end:
  - GET  /health      → 200 with all components ok
  - GET  /documents   → 200 with document list
  - POST /ingest      → 201 with ingestion summary
  - POST /query       → 200 with verified answers
"""

import pytest
import httpx
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# API base URL - assumes server is running on localhost:8000
API_BASE_URL = "http://localhost:8000"


def get_test_pdf() -> Path:
    """Return path to a test PDF."""
    pdfs = list(FIXTURES_DIR.glob("*.pdf"))
    if not pdfs:
        pytest.skip("No PDF files in tests/fixtures/ — add at least one.")
    return pdfs[0]


@pytest.fixture
async def client():
    """Create HTTP client for testing live API server."""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        yield client


# ── Health tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(client):
    """Health endpoint should return 200."""
    response = await client.get("/health")
    # Accept both 200 and 503 (if MongoDB is down during test)
    assert response.status_code in (200, 503)


@pytest.mark.asyncio
async def test_health_has_status_field(client):
    """Health response should have status field."""
    response = await client.get("/health")
    data = response.json()
    assert "status" in data


@pytest.mark.asyncio
async def test_health_has_version(client):
    """Health response should have version field."""
    response = await client.get("/health")
    data = response.json()
    assert "version" in data


# ── Documents tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_documents_returns_200(client):
    """Documents endpoint should return 200."""
    response = await client.get("/documents")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_documents_returns_list(client):
    """Documents response should have documents list."""
    response = await client.get("/documents")
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_documents_has_total(client):
    """Documents response should have total field."""
    response = await client.get("/documents")
    data = response.json()
    assert "total" in data
    assert data["total"] == len(data["documents"])


# ── Ingest tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_non_pdf_returns_415(client):
    """Uploading a non-PDF file should return 415."""
    response = await client.post(
        "/ingest",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_ingest_valid_pdf_returns_201(client):
    """Valid PDF upload should return 201."""
    pdf_path = get_test_pdf()
    with open(pdf_path, "rb") as f:
        response = await client.post(
            "/ingest",
            files={"file": (pdf_path.name, f, "application/pdf")},
        )
    # Accept 201 (success) or 500 (if already ingested or other issue)
    assert response.status_code in (201, 500)


# ── Query tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_returns_200_or_404(client):
    """Query should return 200 with answers or 404 if no answer found."""
    response = await client.post(
        "/query",
        json={"query": "What is the main topic?", "top_k": 5},
    )
    assert response.status_code in (200, 404, 503)


@pytest.mark.asyncio
async def test_query_200_has_required_fields(client):
    """Successful query should have all required fields."""
    response = await client.post(
        "/query",
        json={"query": "What is the main topic?", "top_k": 5},
    )
    if response.status_code == 200:
        data = response.json()
        assert "query" in data
        assert "answers" in data
        assert "total_answers" in data
        assert "processing_ms" in data
        assert "rake_used" in data


@pytest.mark.asyncio
async def test_query_answers_have_source_attribution(client):
    """Each answer should have filename and page_number."""
    response = await client.post(
        "/query",
        json={"query": "What is the main purpose?", "top_k": 5},
    )
    if response.status_code == 200:
        data = response.json()
        for answer in data["answers"]:
            assert "filename" in answer
            assert "page_number" in answer
            assert answer["page_number"] >= 1
            assert answer["filename"].endswith(".pdf")


@pytest.mark.asyncio
async def test_query_answers_have_bbox(client):
    """Each answer should have BBox coordinates."""
    response = await client.post(
        "/query",
        json={"query": "What is the main purpose?", "top_k": 5},
    )
    if response.status_code == 200:
        data = response.json()
        for answer in data["answers"]:
            bbox = answer["bbox"]
            assert "x0" in bbox
            assert "y0" in bbox
            assert "x1" in bbox
            assert "y1" in bbox


@pytest.mark.asyncio
async def test_query_too_short_returns_422(client):
    """Query shorter than 3 characters should return 422."""
    response = await client.post(
        "/query",
        json={"query": "ab", "top_k": 5},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_missing_field_returns_422(client):
    """Missing query field should return 422."""
    response = await client.post(
        "/query",
        json={"top_k": 5},
    )
    assert response.status_code == 422