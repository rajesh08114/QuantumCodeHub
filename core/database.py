"""
Database connection pooling.
"""
import asyncpg
import logging
from core.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


def _pool_is_ready() -> bool:
    if _pool is None:
        return False
    return not _pool._closed  # noqa: SLF001 - asyncpg exposes no public flag.


async def connect_to_db():
    global _pool
    if _pool_is_ready():
        return

    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        ssl="require",
        min_size=1,
        max_size=settings.DB_POOL_SIZE or 10,
    )
    logger.info("Database pool created")


async def close_db():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


async def get_db_connection() -> asyncpg.Connection:
    if not _pool_is_ready():
        await connect_to_db()
    if _pool is None:
        raise RuntimeError("Database pool is not initialized.")
    return await _pool.acquire()


async def release_db_connection(conn: asyncpg.Connection | None):
    if conn is None:
        return
    if _pool is None:
        await conn.close()
        return
    await _pool.release(conn)
