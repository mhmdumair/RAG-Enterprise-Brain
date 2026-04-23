"""
brain/indexer.py
================
FAISS IndexHNSWFlat vector index — build, search, persist, load.

Why HNSW?
    - O(log N) approximate nearest-neighbour search
    - No GPU required
    - Excellent recall at reasonable ef_search values
    - Supports incremental adds (though we rebuild on each ingest)

Index lifecycle:
    1. Build  — create empty index, add vectors
    2. Save   — persist to storage/indexes/brain.index
    3. Load   — restore from disk on API startup
    4. Search — given a query vector, return top-K vector IDs

The integer IDs returned by FAISS correspond directly to
the vector_id field in the MongoDB chunks collection.
"""

import numpy as np
import faiss
from pathlib import Path

from core.config import settings
from core.logger import get_logger
from core.exceptions import IndexError as BrainIndexError

logger = get_logger(__name__)


class FAISSIndex:
    """
    Wrapper around faiss.IndexHNSWFlat.

    Usage:
        index = FAISSIndex()
        index.build(vectors)          # initial build
        index.save()                  # persist to disk
        index.load()                  # restore from disk
        ids, scores = index.search(query_vector, k=5)
    """

    def __init__(self):
        self._dim = settings.embedding_dim
        self._index: faiss.IndexHNSWFlat | None = None
        self._index_path = settings.faiss_index_path
        self._total_vectors = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray) -> None:
        """
        Create a new HNSW index and add all vectors.

        Args:
            vectors — np.ndarray shape (N, 384) dtype float32

        Raises:
            BrainIndexError — if vectors have wrong shape or dtype
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise BrainIndexError(
                "build",
                f"Expected shape (N, {self._dim}), got {vectors.shape}",
            )

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        logger.info(
            "Building FAISS index",
            extra={"vectors": vectors.shape[0], "dim": self._dim},
        )

        try:
            # IndexHNSWFlat stores raw float32 vectors (no quantization)
            # HNSW M = number of links per node (higher = better recall, more RAM)
            index = faiss.IndexHNSWFlat(self._dim, settings.faiss_hnsw_m)
            index.hnsw.efConstruction = settings.faiss_hnsw_ef_construction
            index.hnsw.efSearch = settings.faiss_hnsw_ef_search

            # FAISS assigns sequential integer IDs: 0, 1, 2, ...
            # These IDs become vector_id in MongoDB chunks collection
            index.add(vectors)

            self._index = index
            self._total_vectors = index.ntotal

        except Exception as exc:
            raise BrainIndexError("build", str(exc)) from exc

        logger.info(
            "FAISS index built",
            extra={"total_vectors": self._total_vectors},
        )

    def add(self, vectors: np.ndarray) -> list[int]:
        """
        Add new vectors to an existing index.
        Returns the list of new vector IDs assigned.

        Args:
            vectors — np.ndarray shape (M, 384) dtype float32

        Returns:
            list of integer IDs [old_total, old_total+1, ..., new_total-1]
        """
        if self._index is None:
            raise BrainIndexError("add", "Index not initialized. Call build() first.")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        start_id = self._index.ntotal
        try:
            self._index.add(vectors)
            self._total_vectors = self._index.ntotal
        except Exception as exc:
            raise BrainIndexError("add", str(exc)) from exc

        new_ids = list(range(start_id, self._index.ntotal))
        logger.info(
            "Vectors added to index",
            extra={"added": len(new_ids), "total": self._total_vectors},
        )
        return new_ids

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        Find the top-K nearest neighbours for a query vector.

        Args:
            query_vector — np.ndarray shape (384,) or (1, 384) float32
            k            — number of results (defaults to settings.top_k_chunks)

        Returns:
            (vector_ids, scores)
            vector_ids — list of int (FAISS index positions)
            scores     — list of float (L2 distances, lower = more similar)

        Raises:
            BrainIndexError — if index is empty or query has wrong shape
        """
        if self._index is None or self._total_vectors == 0:
            raise BrainIndexError("search", "Index is empty. Ingest documents first.")

        k = k or settings.top_k_chunks
        k = min(k, self._total_vectors)  # can't retrieve more than we have

        # Ensure shape is (1, dim)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        try:
            distances, indices = self._index.search(query_vector, k)
        except Exception as exc:
            raise BrainIndexError("search", str(exc)) from exc

        # FAISS returns -1 for unfilled slots — filter those out
        vector_ids = [int(i) for i in indices[0] if i != -1]
        scores = [float(d) for d, i in zip(distances[0], indices[0]) if i != -1]

        return vector_ids, scores

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> None:
        """
        Persist the FAISS index to disk.

        Args:
            path — override the default storage/indexes/brain.index path
        """
        if self._index is None:
            raise BrainIndexError("save", "No index to save.")

        save_path = path or self._index_path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(self._index, str(save_path))
        except Exception as exc:
            raise BrainIndexError("save", str(exc)) from exc

        logger.info(
            "FAISS index saved",
            extra={"path": str(save_path), "vectors": self._total_vectors},
        )

    def load(self, path: Path | None = None) -> bool:
        """
        Load a persisted FAISS index from disk.

        Returns:
            True if loaded successfully, False if file doesn't exist.

        Raises:
            BrainIndexError — if file exists but is corrupt
        """
        load_path = path or self._index_path
        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(
                "No FAISS index file found",
                extra={"path": str(load_path)},
            )
            return False

        try:
            self._index = faiss.read_index(str(load_path))
            self._total_vectors = self._index.ntotal
        except Exception as exc:
            raise BrainIndexError("load", str(exc)) from exc

        logger.info(
            "FAISS index loaded",
            extra={"path": str(load_path), "vectors": self._total_vectors},
        )
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._total_vectors

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._total_vectors > 0