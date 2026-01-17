"""Market Data Service for fetching and caching real OHLCV candle data.

This service provides:
- Real candlestick data from WEEX API (not just ticker polling)
- Multi-timeframe support (1m, 5m, 15m, 1H, 4H, 1D)
- In-memory caching with Redis persistence
- Background refresh task for near-real-time data
- Housekeeping for memory management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from weex_client import WeexClient
    from .redis_client import RedisClient


class MarketDataService:
    """Service for fetching and caching OHLCV candle data.

    Supports multiple timeframes and provides cached access to candle data
    for use by strategy agents and indicator calculations.

    Features:
    - In-memory cache for fast access
    - Redis persistence for data survival across restarts
    - Automatic background refresh
    - Memory housekeeping with configurable limits
    """

    # Supported timeframes mapped to WEEX API format
    # WEEX contract API uses: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 1w, 1M
    TIMEFRAMES = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
        "1M": "1M",
    }

    # How often to refresh each timeframe (in seconds)
    REFRESH_INTERVALS = {
        "1m": 60,      # Refresh every minute
        "5m": 300,     # Refresh every 5 minutes
        "15m": 900,    # Refresh every 15 minutes
        "30m": 1800,   # Refresh every 30 minutes
        "1h": 3600,    # Refresh every hour
        "4h": 14400,   # Refresh every 4 hours
        "12h": 43200,  # Refresh every 12 hours
        "1d": 86400,   # Refresh daily
        "1w": 604800,  # Refresh weekly
    }

    # Cache limits per timeframe (based on strategy requirements)
    # Turtle Trading needs 55+ daily, others need ~50 for EMA calculations
    CACHE_LIMITS = {
        "1m": 100,   # Only for aggregation
        "5m": 100,   # Entry timing
        "15m": 100,  # Short-term signals
        "30m": 100,  # Medium signals
        "1h": 100,   # Trend Following, Mean Reversion
        "4h": 100,   # Mean Reversion HTF, Alpha Generator
        "12h": 100,  # Longer-term analysis
        "1d": 100,   # Turtle Trading (needs 55+), Regime
        "1w": 50,    # Weekly analysis
    }

    # Redis TTL per timeframe (keep data longer than refresh interval)
    REDIS_TTL = {
        "1m": 3600,      # 1 hour
        "5m": 7200,      # 2 hours
        "15m": 14400,    # 4 hours
        "30m": 28800,    # 8 hours
        "1h": 86400,     # 1 day
        "4h": 259200,    # 3 days
        "12h": 432000,   # 5 days
        "1d": 604800,    # 7 days
        "1w": 2592000,   # 30 days
    }

    # Redis key patterns
    REDIS_KEY_CANDLES = "candles:{symbol}:{timeframe}"
    REDIS_KEY_REFRESH = "refresh:{symbol}:{timeframe}"

    def __init__(
        self,
        weex_client: "WeexClient",
        redis_client: Optional["RedisClient"] = None,
        symbols: Optional[List[str]] = None,
        default_timeframes: Optional[List[str]] = None,
        cache_size: int = 200,
    ):
        """Initialize the Market Data Service.

        Args:
            weex_client: WEEX API client instance
            redis_client: Redis client for persistence (optional)
            symbols: List of symbols to track (default: ["BTCUSDT"])
            default_timeframes: Timeframes to pre-fetch (default: ["1h", "4h", "1d"])
            cache_size: Number of candles to fetch from API per request
        """
        self.weex_client = weex_client
        self.redis = redis_client
        self.symbols = symbols or ["BTCUSDT"]
        self.default_timeframes = default_timeframes or ["1h", "4h", "1d"]
        self.cache_size = cache_size

        # Cache structure: {symbol: {timeframe: DataFrame}}
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._last_refresh: Dict[str, Dict[str, datetime]] = {}

        # Background tasks
        self._refresh_task: Optional[asyncio.Task] = None
        self._housekeeping_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the market data service and begin background tasks."""
        logger.info("Starting Market Data Service...")
        self._running = True

        # Initial fetch for all symbols and timeframes
        for symbol in self.symbols:
            self._cache[symbol] = {}
            self._last_refresh[symbol] = {}

            for tf in self.default_timeframes:
                try:
                    # Try loading from Redis first
                    loaded_from_redis = await self._load_from_redis(symbol, tf)

                    if loaded_from_redis:
                        logger.info(f"Loaded {symbol} {tf} from Redis: {len(self._cache[symbol].get(tf, []))} candles")
                    else:
                        # Fetch from API
                        await self._fetch_and_cache(symbol, tf)
                        logger.info(f"Fetched {symbol} {tf} from API: {len(self._cache[symbol].get(tf, []))} candles")

                except Exception as e:
                    logger.error(f"Failed to initialize {symbol} {tf}: {e}")

        # Start background tasks
        self._refresh_task = asyncio.create_task(self._background_refresh())
        self._housekeeping_task = asyncio.create_task(self._housekeeping_loop())

        logger.info("Market Data Service started")

    async def stop(self):
        """Stop the market data service."""
        logger.info("Stopping Market Data Service...")
        self._running = False

        # Cancel background tasks
        for task in [self._refresh_task, self._housekeeping_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Final save to Redis before stopping
        if self.redis and self.redis.is_connected:
            for symbol in self._cache:
                for tf in self._cache[symbol]:
                    try:
                        await self._save_to_redis(symbol, tf)
                    except Exception as e:
                        logger.error(f"Failed to save {symbol} {tf} to Redis on shutdown: {e}")

        logger.info("Market Data Service stopped")

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Get OHLCV candle data for a symbol and timeframe.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
            limit: Number of candles to return (default: all cached)
            refresh: Force refresh from API

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Normalize timeframe
        tf = timeframe.lower()
        if tf not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use: {list(self.TIMEFRAMES.keys())}")

        # Check if we need to refresh
        if refresh or symbol not in self._cache or tf not in self._cache.get(symbol, {}):
            await self._fetch_and_cache(symbol, tf)
        else:
            # Check if cache is stale
            last_refresh = self._last_refresh.get(symbol, {}).get(tf)
            if last_refresh:
                stale_threshold = timedelta(seconds=self.REFRESH_INTERVALS.get(tf, 3600))
                if datetime.now() - last_refresh > stale_threshold:
                    await self._fetch_and_cache(symbol, tf)

        # Get cached data
        df = self._cache.get(symbol, {}).get(tf, pd.DataFrame())

        if df.empty:
            logger.warning(f"No candle data available for {symbol} {tf}")
            return df

        # Apply limit
        if limit and limit < len(df):
            df = df.tail(limit).reset_index(drop=True)

        return df.copy()

    async def get_latest_candle(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent candle for a symbol and timeframe.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe

        Returns:
            Dictionary with candle data or None
        """
        df = await self.get_candles(symbol, timeframe, limit=1)

        if df.empty:
            return None

        row = df.iloc[-1]
        return {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }

    async def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        limit: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """Get candle data for multiple timeframes.

        Useful for multi-timeframe analysis where you need to confirm
        signals across different time horizons.

        Args:
            symbol: Trading pair
            timeframes: List of timeframes (default: ["1h", "4h", "1d"])
            limit: Number of candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        tfs = timeframes or self.default_timeframes
        result = {}

        for tf in tfs:
            try:
                result[tf] = await self.get_candles(symbol, tf, limit=limit)
            except Exception as e:
                logger.error(f"Failed to get {symbol} {tf}: {e}")
                result[tf] = pd.DataFrame()

        return result

    def get_cached_ohlcv_arrays(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get cached OHLCV data as numpy arrays.

        Convenient for agents that work directly with numpy arrays
        instead of DataFrames.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe

        Returns:
            Dictionary with opens, highs, lows, closes, volumes as arrays
        """
        df = self._cache.get(symbol, {}).get(timeframe.lower())

        if df is None or df.empty:
            return None

        return {
            "timestamps": df["timestamp"].values,
            "opens": df["open"].values.astype(float),
            "highs": df["high"].values.astype(float),
            "lows": df["low"].values.astype(float),
            "closes": df["close"].values.astype(float),
            "volumes": df["volume"].values.astype(float),
        }

    # ==================== Redis Operations ====================

    async def _load_from_redis(self, symbol: str, timeframe: str) -> bool:
        """Load candle data from Redis cache.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe

        Returns:
            True if data was loaded successfully
        """
        if not self.redis or not self.redis.is_connected:
            return False

        try:
            key = self.REDIS_KEY_CANDLES.format(symbol=symbol, timeframe=timeframe)
            data = await self.redis.client.get(key)

            if not data:
                return False

            # Parse JSON to DataFrame
            records = json.loads(data)
            if not records:
                return False

            df = pd.DataFrame(records)

            # Validate minimum data
            min_candles = self.CACHE_LIMITS.get(timeframe, 50) // 2  # At least half required
            if len(df) < min_candles:
                logger.debug(f"Redis cache too small for {symbol} {timeframe}: {len(df)} < {min_candles}")
                return False

            # Ensure symbol is in cache dict
            if symbol not in self._cache:
                self._cache[symbol] = {}
                self._last_refresh[symbol] = {}

            # Store in memory cache
            self._cache[symbol][timeframe] = df
            self._last_refresh[symbol][timeframe] = datetime.now()

            logger.debug(f"Loaded {len(df)} candles from Redis for {symbol} {timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error loading from Redis for {symbol} {timeframe}: {e}")
            return False

    async def _save_to_redis(self, symbol: str, timeframe: str):
        """Save candle data to Redis cache.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
        """
        if not self.redis or not self.redis.is_connected:
            return

        try:
            df = self._cache.get(symbol, {}).get(timeframe)
            if df is None or df.empty:
                return

            key = self.REDIS_KEY_CANDLES.format(symbol=symbol, timeframe=timeframe)

            # Convert to JSON
            data = df.to_json(orient="records")

            # Store with TTL
            ttl = self.REDIS_TTL.get(timeframe, 86400)
            await self.redis.client.setex(key, ttl, data)

            # Update refresh timestamp in Redis
            refresh_key = self.REDIS_KEY_REFRESH.format(symbol=symbol, timeframe=timeframe)
            await self.redis.client.setex(refresh_key, ttl, datetime.now().isoformat())

            logger.debug(f"Saved {len(df)} candles to Redis for {symbol} {timeframe} (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Error saving to Redis for {symbol} {timeframe}: {e}")

    # ==================== Fetch and Cache ====================

    async def _fetch_and_cache(self, symbol: str, timeframe: str):
        """Fetch candle data from WEEX API and cache it.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe (our format, e.g., "1h")
        """
        # Convert to WEEX API format
        weex_interval = self.TIMEFRAMES.get(timeframe)
        if not weex_interval:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        try:
            # Fetch from WEEX API
            klines = await self.weex_client.get_klines(
                symbol=symbol,
                interval=weex_interval,
                limit=self.cache_size,
            )

            if not klines:
                logger.warning(f"No klines returned for {symbol} {timeframe}")
                return

            # Parse to DataFrame
            df = self._parse_klines_to_dataframe(klines)

            # Ensure symbol is in cache dict
            if symbol not in self._cache:
                self._cache[symbol] = {}
                self._last_refresh[symbol] = {}

            # Store in memory cache
            self._cache[symbol][timeframe] = df
            self._last_refresh[symbol][timeframe] = datetime.now()

            logger.debug(f"Cached {len(df)} candles for {symbol} {timeframe}")

            # Persist to Redis (non-blocking)
            asyncio.create_task(self._save_to_redis(symbol, timeframe))

        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            raise

    def _parse_klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """Parse WEEX klines response to pandas DataFrame.

        WEEX returns: [timestamp, open, high, low, close, volume, quote_volume]

        Args:
            klines: Raw klines data from WEEX API

        Returns:
            DataFrame with standardized columns
        """
        if not klines:
            return pd.DataFrame()

        # WEEX klines format: [timestamp, open, high, low, close, volume, quote_volume]
        data = []
        for k in klines:
            if len(k) >= 6:
                data.append({
                    "timestamp": int(k[0]) if k[0] else 0,
                    "open": float(k[1]) if k[1] else 0.0,
                    "high": float(k[2]) if k[2] else 0.0,
                    "low": float(k[3]) if k[3] else 0.0,
                    "close": float(k[4]) if k[4] else 0.0,
                    "volume": float(k[5]) if k[5] else 0.0,
                })

        df = pd.DataFrame(data)

        # Sort by timestamp (oldest first)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # ==================== Background Tasks ====================

    async def _background_refresh(self):
        """Background task to refresh candle data periodically."""
        logger.info("Starting background refresh task")

        # Track when each symbol/timeframe was last refreshed
        next_refresh: Dict[str, Dict[str, datetime]] = {}

        while self._running:
            try:
                now = datetime.now()

                for symbol in self.symbols:
                    if symbol not in next_refresh:
                        next_refresh[symbol] = {}

                    for tf in self.default_timeframes:
                        # Check if it's time to refresh this timeframe
                        scheduled = next_refresh.get(symbol, {}).get(tf)
                        if scheduled and now < scheduled:
                            continue

                        try:
                            await self._fetch_and_cache(symbol, tf)

                            # Schedule next refresh
                            interval = self.REFRESH_INTERVALS.get(tf, 3600)
                            next_refresh[symbol][tf] = now + timedelta(seconds=interval)

                        except Exception as e:
                            logger.error(f"Background refresh failed for {symbol} {tf}: {e}")
                            # Retry in 1 minute on error
                            next_refresh[symbol][tf] = now + timedelta(seconds=60)

                # Sleep for a bit before checking again
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background refresh error: {e}")
                await asyncio.sleep(30)

        logger.info("Background refresh task stopped")

    async def _housekeeping_loop(self):
        """Background task for memory management and cache cleanup."""
        logger.info("Starting housekeeping task")

        while self._running:
            try:
                # Run every 5 minutes
                await asyncio.sleep(300)

                # Trim memory cache to limits
                self._trim_memory_cache()

                # Log memory usage
                self._log_memory_usage()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Housekeeping error: {e}")
                await asyncio.sleep(60)

        logger.info("Housekeeping task stopped")

    def _trim_memory_cache(self):
        """Trim each cache to its limit (keep most recent candles)."""
        trimmed = 0
        for symbol in self._cache:
            for tf in self._cache[symbol]:
                limit = self.CACHE_LIMITS.get(tf, 100)
                df = self._cache[symbol][tf]
                if not df.empty and len(df) > limit:
                    self._cache[symbol][tf] = df.tail(limit).reset_index(drop=True)
                    trimmed += len(df) - limit

        if trimmed > 0:
            logger.debug(f"Housekeeping: trimmed {trimmed} candles from cache")

    def _log_memory_usage(self):
        """Log current memory usage of the cache."""
        total_bytes = 0
        for symbol in self._cache:
            for tf in self._cache[symbol]:
                df = self._cache[symbol][tf]
                if not df.empty:
                    total_bytes += df.memory_usage(deep=True).sum()

        logger.info(f"Cache memory usage: {total_bytes / 1024:.1f} KB")

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Dictionary with comprehensive cache statistics
        """
        total_bytes = 0
        stats = {
            "running": self._running,
            "redis_connected": self.redis.is_connected if self.redis else False,
            "symbols": self.symbols,
            "timeframes": self.default_timeframes,
            "memory_kb": 0,
            "cache": {},
        }

        for symbol in self._cache:
            stats["cache"][symbol] = {}
            for tf in self._cache[symbol]:
                df = self._cache[symbol][tf]
                bytes_used = df.memory_usage(deep=True).sum() if not df.empty else 0
                total_bytes += bytes_used

                # Get last candle timestamp
                last_ts = None
                if not df.empty:
                    last_ts = int(df["timestamp"].iloc[-1])

                stats["cache"][symbol][tf] = {
                    "candles": len(df) if not df.empty else 0,
                    "limit": self.CACHE_LIMITS.get(tf, 100),
                    "memory_kb": round(bytes_used / 1024, 2),
                    "last_candle_ts": last_ts,
                    "last_refresh": str(self._last_refresh.get(symbol, {}).get(tf, "never")),
                }

        stats["memory_kb"] = round(total_bytes / 1024, 2)

        return stats

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get service statistics including Redis info.

        Returns:
            Dictionary with comprehensive cache and Redis statistics
        """
        stats = self.get_stats()

        # Add Redis stats if connected
        if self.redis and self.redis.is_connected:
            try:
                info = await self.redis.get_info("memory")
                stats["redis_memory_mb"] = round(info.get("used_memory", 0) / 1024 / 1024, 2)
                stats["redis_peak_mb"] = round(info.get("used_memory_peak", 0) / 1024 / 1024, 2)
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")

        return stats
