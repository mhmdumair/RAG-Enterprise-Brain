"""
brain/embedder.py
=================
Bi-encoder embedding using sentence-transformers/all-MiniLM-L6-v2.

Converts text chunks into 384-dimensional float32 vectors.
These vectors are stored in the FAISS index and used for
semantic similarity search at query time.

Key properties:
  - Model is loaded once and reused (singleton pattern)
  - Batch processing for efficiency
  - Vectors are L2-normalized (unit length) for cosine similarity
  - Model weights cached in storage/models/ (offline after first run)

Flow:
    list[TextChunk] → encode in batches → numpy array (N x 384)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from core.config import settings
from core.logger import get_logger
from core.exceptions import EmbeddingError
from brain.chunker import TextChunk

logger = get_logger(__name__)


class Embedder:
    """
    Wraps SentenceTransformer for batch chunk embedding.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(chunks)   # returns np.ndarray shape (N, 384)
    """

    def __init__(self):
        logger.info(
            "Loading embedding model",
            extra={"model": settings.embedding_model_name},
        )
        try:
            self._model = SentenceTransformer(
                settings.embedding_model_name,
                cache_folder=str(settings.model_cache_dir),
            )
            # Verify output dimension matches config
            test_vec = self._model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_vec.shape[1]
            if actual_dim != settings.embedding_dim:
                raise EmbeddingError(
                    f"Model output dim {actual_dim} != config dim {settings.embedding_dim}"
                )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Failed to load model: {exc}") from exc

        logger.info(
            "Embedding model loaded",
            extra={
                "model": settings.embedding_model_name,
                "dim": settings.embedding_dim,
            },
        )

    def embed(
        self,
        chunks: list[TextChunk],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of TextChunks into a 2D numpy array of vectors.

        Args:
            chunks        — list of TextChunk objects
            batch_size    — number of chunks per encoding batch
            show_progress — show tqdm progress bar (useful for large docs)

        Returns:
            np.ndarray of shape (len(chunks), 384) dtype float32
            Vectors are L2-normalized (unit length).

        Raises:
            EmbeddingError — if encoding fails
        """
        if not chunks:
            return np.empty((0, settings.embedding_dim), dtype=np.float32)

        texts = [chunk.text for chunk in chunks]

        logger.info(
            "Embedding chunks",
            extra={"count": len(texts), "batch_size": batch_size},
        )

        try:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2 normalize → cosine via dot product
                show_progress_bar=show_progress,
            )
        except Exception as exc:
            raise EmbeddingError(f"Encoding failed: {exc}") from exc

        vectors = vectors.astype(np.float32)

        logger.info(
            "Chunks embedded",
            extra={"shape": list(vectors.shape)},
        )

        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a 1D vector.
        Used at query time by the retriever.

        Args:
            query — the audit question string

        Returns:
            np.ndarray of shape (384,) dtype float32, L2-normalized.
        """
        if not query or not query.strip():
            raise EmbeddingError("Query string is empty.")

        try:
            vector = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vector[0].astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc