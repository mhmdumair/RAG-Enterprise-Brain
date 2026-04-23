"""
core/config.py
==============
Single source of truth for all configuration.
Every module imports from here — never from os.environ directly.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── MongoDB ───────────────────────────────────────────────────────────────
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_db_name: str = Field(default="enterprise_brain")

    # ── Collections ───────────────────────────────────────────────────────────
    collection_documents: str = "documents"
    collection_chunks: str = "chunks"
    collection_results: str = "results"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_title: str = "Enterprise Brain API"
    api_version: str = "0.1.0"

    # ── Storage paths ─────────────────────────────────────────────────────────
    upload_dir: Path = Field(default=BASE_DIR / "storage" / "uploads")
    faiss_index_path: Path = Field(default=BASE_DIR / "storage" / "indexes" / "brain.index")
    model_cache_dir: Path = Field(default=BASE_DIR / "storage" / "models")

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # ── QA model ──────────────────────────────────────────────────────────────
    qa_model_name: str = Field(default="deepset/roberta-base-squad2")
    qa_max_answer_len: int = Field(default=100)
    qa_max_seq_len: int = Field(default=512)
    qa_doc_stride: int = Field(default=128)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_chunks: int = Field(default=5)

    # ── Abstention threshold ──────────────────────────────────────────────────
    tau_ans: float = Field(default=0.1)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=400)
    chunk_overlap: int = Field(default=80)

    # ── Ingestion limits ──────────────────────────────────────────────────────
    max_pdfs: int = Field(default=10)
    max_pages_per_pdf: int = Field(default=20)
    max_file_size_mb: int = Field(default=50)

    # ── FAISS HNSW ────────────────────────────────────────────────────────────
    faiss_hnsw_m: int = Field(default=32)
    faiss_hnsw_ef_construction: int = Field(default=200)
    faiss_hnsw_ef_search: int = Field(default=128)

    # ── RAKE fallback ─────────────────────────────────────────────────────────
    rake_max_words: int = Field(default=3)
    rake_top_n: int = Field(default=5)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    def ensure_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


# ── Singleton — import this everywhere ───────────────────────────────────────
settings = Settings()
settings.ensure_dirs()