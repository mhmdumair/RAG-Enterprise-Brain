"""
db/client.py
============
Async MongoDB client — singleton pattern.
The connection is opened once at startup and reused everywhere.

Usage:
    from db.client import get_database
    db = await get_database()
    collection = db["chunks"]
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton — one client for the entire process lifetime
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """
    Return the Motor client, creating it on first call.
    Thread-safe for async use — Motor manages its own connection pool.
    """
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=5000,   # fail fast if MongoDB is down
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            maxPoolSize=10,
            minPoolSize=1,
        )
        logger.info(
            "MongoDB client created",
            extra={"uri": settings.mongo_uri, "db": settings.mongo_db_name},
        )
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """
    Return the enterprise_brain database handle.
    Call this inside any async route or service function.
    """
    return get_client()[settings.mongo_db_name]


async def ping_database() -> bool:
    """
    Check that MongoDB is reachable.
    Used by the /health endpoint.

    Returns:
        True if ping succeeds, False otherwise.
    """
    try:
        client = get_client()
        await client.admin.command("ping")
        logger.info("MongoDB ping successful")
        return True
    except Exception as exc:
        logger.error("MongoDB ping failed", extra={"error": str(exc)})
        return False


async def close_client() -> None:
    """
    Cleanly close the MongoDB connection.
    Called during FastAPI shutdown lifespan event.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed")