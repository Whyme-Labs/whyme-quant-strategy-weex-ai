"""Redis client wrapper for async operations with connection pooling.

Provides a simple interface for Redis operations used by the Market Data Service
for caching OHLCV candle data with persistence across container restarts.
"""

import redis.asyncio as redis
from typing import Optional
from loguru import logger


class RedisClient:
    """Async Redis client wrapper with connection pooling.

    Features:
    - Connection pooling for efficient resource usage
    - Async operations for non-blocking I/O
    - Automatic reconnection handling
    - Health check via ping
    """

    def __init__(self, url: str = "redis://localhost:6379"):
        """Initialize Redis client.

        Args:
            url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.url = url
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self) -> bool:
        """Initialize connection pool and verify connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=10,
                decode_responses=False,  # We handle encoding for binary data
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Verify connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Redis connected: {self.url}")
            return True

        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            self._connected = False
            return False
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            self._connected = False
            return False

    async def close(self):
        """Close connections and cleanup."""
        try:
            if self._client:
                await self._client.close()
            if self._pool:
                await self._pool.disconnect()
            self._connected = False
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    async def ping(self) -> bool:
        """Check if Redis is responsive.

        Returns:
            True if Redis responds to ping
        """
        try:
            if self._client:
                await self._client.ping()
                return True
        except Exception:
            pass
        return False

    async def get_info(self, section: str = "memory") -> dict:
        """Get Redis server info.

        Args:
            section: Info section (memory, stats, replication, etc.)

        Returns:
            Dictionary with Redis info
        """
        try:
            if self._client:
                return await self._client.info(section)
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
        return {}

    @property
    def client(self) -> Optional[redis.Redis]:
        """Get the underlying Redis client for direct operations."""
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None
