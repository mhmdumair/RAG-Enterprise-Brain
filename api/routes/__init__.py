"""
api/routes/__init__.py
======================
Route module exports.
"""

from api.routes import health, documents, ingest, query

__all__ = ["health", "documents", "ingest", "query"]