"""
tests/test_embedder.py
======================
Tests for brain/embedder.py

Covers:
  - embed() returns correct shape (N, 384)
  - embed() returns float32 dtype
  - Vectors are L2-normalized (unit length)
  - embed_query() returns shape (384,)
  - Empty chunk list returns empty array
  - Empty query raises EmbeddingError
  - Batch processing produces consistent results
"""

import pytest
import numpy as np

from brain.embedder import Embedder
from brain.chunker import TextChunk
from core.exceptions import EmbeddingError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedder():
    """Shared embedder — loaded once for the entire module."""
    return Embedder()


def make_fake_chunk(text: str, index: int = 0) -> TextChunk:
    """Create a minimal TextChunk for testing."""
    return TextChunk(
        chunk_id=f"test_chunk_{index}",
        document_id="test_doc",
        filename="test.pdf",
        page_number=1,
        chunk_index=index,
        text=text,
        bbox={"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0,
              "page_width": 595.0, "page_height": 842.0},
        char_start=0,
        char_end=len(text),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_returns_correct_shape(self, embedder):
        """embed() should return (N, 384) array."""
        chunks = [make_fake_chunk(f"Sample text number {i}") for i in range(5)]
        vectors = embedder.embed(chunks)
        assert vectors.shape == (5, 384)

    def test_embed_returns_float32(self, embedder):
        """embed() should return float32 dtype."""
        chunks = [make_fake_chunk("Test text")]
        vectors = embedder.embed(chunks)
        assert vectors.dtype == np.float32

    def test_vectors_are_normalized(self, embedder):
        """L2 norms of all vectors should be ~1.0 (unit length)."""
        chunks = [make_fake_chunk(f"Sentence number {i}") for i in range(10)]
        vectors = embedder.embed(chunks)
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_empty_list_returns_empty(self, embedder):
        """embed() with empty input should return empty array."""
        vectors = embedder.embed([])
        assert vectors.shape == (0, 384)

    def test_embed_query_returns_correct_shape(self, embedder):
        """embed_query() should return shape (384,)."""
        vector = embedder.embed_query("What is the warranty period?")
        assert vector.shape == (384,)

    def test_embed_query_returns_float32(self, embedder):
        """embed_query() should return float32 dtype."""
        vector = embedder.embed_query("Test query")
        assert vector.dtype == np.float32

    def test_embed_query_is_normalized(self, embedder):
        """Query vector should be L2-normalized."""
        vector = embedder.embed_query("What is the main topic?")
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_query_empty_raises(self, embedder):
        """Empty query string should raise EmbeddingError."""
        with pytest.raises(EmbeddingError):
            embedder.embed_query("")

    def test_embed_query_whitespace_raises(self, embedder):
        """Whitespace-only query should raise EmbeddingError."""
        with pytest.raises(EmbeddingError):
            embedder.embed_query("   ")

    def test_similar_texts_have_close_vectors(self, embedder):
        """Semantically similar texts should produce similar vectors."""
        chunks_a = [make_fake_chunk("The warranty is two years.")]
        chunks_b = [make_fake_chunk("The guarantee period is 24 months.")]
        chunks_c = [make_fake_chunk("The sky is blue and the ocean is vast.")]

        vec_a = embedder.embed(chunks_a)[0]
        vec_b = embedder.embed(chunks_b)[0]
        vec_c = embedder.embed(chunks_c)[0]

        # Similar sentences should have higher dot product than dissimilar
        sim_ab = float(np.dot(vec_a, vec_b))
        sim_ac = float(np.dot(vec_a, vec_c))
        assert sim_ab > sim_ac, "Similar sentences should be closer than dissimilar ones"

    def test_batch_consistency(self, embedder):
        """Same text embedded in different batch positions should give same vector."""
        text = "Consistent embedding test sentence."
        chunks = [make_fake_chunk(text, i) for i in range(3)]
        vectors = embedder.embed(chunks)
        # All 3 vectors should be identical (same text)
        np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-5)
        np.testing.assert_allclose(vectors[1], vectors[2], atol=1e-5)