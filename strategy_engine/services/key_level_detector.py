"""Key Level Detector Service.

Comprehensive support and resistance detection including:
1. Dynamic Swing High/Low Detection
2. Fibonacci Retracement/Extension Levels
3. Level Strength Tracking (test count, bounce rate)
4. Multi-timeframe S/R Clustering
5. Volume Profile S/R Detection
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from .market_data_service import MarketDataService
    from .redis_client import RedisClient


class LevelType(Enum):
    """Type of support/resistance level."""
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    FIBONACCI = "fibonacci"
    VOLUME_NODE = "volume_node"
    PIVOT = "pivot"
    CLUSTER = "cluster"


class LevelStrength(Enum):
    """Strength of a support/resistance level."""
    WEAK = "weak"  # 1-2 tests
    MODERATE = "moderate"  # 3-4 tests
    STRONG = "strong"  # 5+ tests
    VERY_STRONG = "very_strong"  # 7+ tests with high bounce rate


@dataclass
class KeyLevel:
    """A support or resistance level."""
    price: float
    level_type: LevelType
    is_support: bool  # True = support, False = resistance
    strength: LevelStrength
    timeframe: str
    test_count: int = 0
    bounce_count: int = 0
    break_count: int = 0
    last_test: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def bounce_rate(self) -> float:
        """Calculate bounce rate (successful holds)."""
        total = self.bounce_count + self.break_count
        return self.bounce_count / total if total > 0 else 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": self.price,
            "level_type": self.level_type.value,
            "is_support": self.is_support,
            "strength": self.strength.value,
            "timeframe": self.timeframe,
            "test_count": self.test_count,
            "bounce_count": self.bounce_count,
            "break_count": self.break_count,
            "bounce_rate": self.bounce_rate,
            "last_test": self.last_test.isoformat() if self.last_test else None,
        }


@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels."""
    swing_high: float
    swing_low: float
    direction: str  # "up" (retracement from high) or "down" (retracement from low)

    # Retracement levels (0-100% of the move)
    fib_236: float = 0.0
    fib_382: float = 0.0
    fib_500: float = 0.0
    fib_618: float = 0.0
    fib_786: float = 0.0

    # Extension levels (beyond 100%)
    fib_1272: float = 0.0
    fib_1618: float = 0.0
    fib_2618: float = 0.0

    def __post_init__(self):
        """Calculate Fibonacci levels."""
        range_size = self.swing_high - self.swing_low

        if self.direction == "up":
            # Retracement from high (price pulling back)
            self.fib_236 = self.swing_high - (0.236 * range_size)
            self.fib_382 = self.swing_high - (0.382 * range_size)
            self.fib_500 = self.swing_high - (0.500 * range_size)
            self.fib_618 = self.swing_high - (0.618 * range_size)
            self.fib_786 = self.swing_high - (0.786 * range_size)
            # Extensions below the low
            self.fib_1272 = self.swing_low - (0.272 * range_size)
            self.fib_1618 = self.swing_low - (0.618 * range_size)
            self.fib_2618 = self.swing_low - (1.618 * range_size)
        else:
            # Retracement from low (price bouncing up)
            self.fib_236 = self.swing_low + (0.236 * range_size)
            self.fib_382 = self.swing_low + (0.382 * range_size)
            self.fib_500 = self.swing_low + (0.500 * range_size)
            self.fib_618 = self.swing_low + (0.618 * range_size)
            self.fib_786 = self.swing_low + (0.786 * range_size)
            # Extensions above the high
            self.fib_1272 = self.swing_high + (0.272 * range_size)
            self.fib_1618 = self.swing_high + (0.618 * range_size)
            self.fib_2618 = self.swing_high + (1.618 * range_size)

    def get_all_levels(self) -> Dict[str, float]:
        """Get all Fibonacci levels as a dictionary."""
        return {
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "fib_23.6%": self.fib_236,
            "fib_38.2%": self.fib_382,
            "fib_50.0%": self.fib_500,
            "fib_61.8%": self.fib_618,
            "fib_78.6%": self.fib_786,
            "fib_127.2%": self.fib_1272,
            "fib_161.8%": self.fib_1618,
            "fib_261.8%": self.fib_2618,
        }


@dataclass
class VolumeNode:
    """High Volume Node from Volume Profile."""
    price: float
    volume: float
    is_hvn: bool  # High Volume Node (support/resistance)
    is_lvn: bool  # Low Volume Node (price moves fast through)
    price_range: Tuple[float, float]  # Price range this node covers


class KeyLevelDetector:
    """Comprehensive key level detection service.

    Provides:
    - Swing high/low detection across timeframes
    - Fibonacci retracement and extension levels
    - Level strength tracking with bounce/break statistics
    - Multi-timeframe level clustering
    - Volume Profile based support/resistance
    """

    # Fibonacci ratios
    FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]

    def __init__(
        self,
        market_data_service: Optional["MarketDataService"] = None,
        redis_client: Optional["RedisClient"] = None,
        swing_lookback: int = 5,  # Candles to look back for swing detection
        cluster_threshold: float = 0.005,  # 0.5% for clustering nearby levels
        volume_profile_bins: int = 50,  # Number of price bins for volume profile
    ):
        """Initialize KeyLevelDetector.

        Args:
            market_data_service: For fetching candle data
            redis_client: For persisting level data
            swing_lookback: Number of candles for swing high/low detection
            cluster_threshold: Percentage threshold for clustering nearby levels
            volume_profile_bins: Number of bins for volume profile analysis
        """
        self.market_data_service = market_data_service
        self.redis_client = redis_client
        self.swing_lookback = swing_lookback
        self.cluster_threshold = cluster_threshold
        self.volume_profile_bins = volume_profile_bins

        # Cache for levels by symbol
        self._levels_cache: Dict[str, List[KeyLevel]] = {}
        self._fib_cache: Dict[str, FibonacciLevels] = {}
        self._volume_nodes_cache: Dict[str, List[VolumeNode]] = {}

    async def detect_all_levels(
        self,
        symbol: str,
        candles_1h: List[Dict] = None,
        candles_4h: List[Dict] = None,
        candles_1d: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Detect all key levels for a symbol.

        Args:
            symbol: Trading symbol
            candles_1h: 1-hour candle data
            candles_4h: 4-hour candle data
            candles_1d: Daily candle data

        Returns:
            Dictionary with all detected levels
        """
        all_levels: List[KeyLevel] = []

        # 1. Detect swing highs/lows across timeframes
        if candles_1h and len(candles_1h) >= self.swing_lookback * 2:
            swing_levels_1h = self._detect_swing_levels(candles_1h, "1h")
            all_levels.extend(swing_levels_1h)

        if candles_4h and len(candles_4h) >= self.swing_lookback * 2:
            swing_levels_4h = self._detect_swing_levels(candles_4h, "4h")
            all_levels.extend(swing_levels_4h)

        if candles_1d and len(candles_1d) >= self.swing_lookback * 2:
            swing_levels_1d = self._detect_swing_levels(candles_1d, "1d")
            all_levels.extend(swing_levels_1d)

        # 2. Calculate Fibonacci levels from significant swings
        fib_levels = None
        if candles_1d and len(candles_1d) >= 20:
            fib_levels = self._calculate_fibonacci_levels(candles_1d)
            if fib_levels:
                all_levels.extend(self._fib_to_key_levels(fib_levels, "1d"))
                self._fib_cache[symbol] = fib_levels

        # 3. Detect volume profile nodes
        volume_nodes = None
        if candles_4h and len(candles_4h) >= 50:
            volume_nodes = self._calculate_volume_profile(candles_4h)
            if volume_nodes:
                all_levels.extend(self._volume_nodes_to_key_levels(volume_nodes, "4h"))
                self._volume_nodes_cache[symbol] = volume_nodes

        # 4. Cluster nearby levels
        clustered_levels = self._cluster_levels(all_levels)

        # 5. Calculate level strength
        current_price = None
        if candles_1h:
            current_price = float(candles_1h[-1].get("close", 0))

        if current_price:
            self._update_level_strength(clustered_levels, current_price)

        # Cache the levels
        self._levels_cache[symbol] = clustered_levels

        # Persist to Redis if available
        if self.redis_client:
            await self._persist_levels(symbol, clustered_levels)

        return {
            "levels": [l.to_dict() for l in clustered_levels],
            "fibonacci": fib_levels.get_all_levels() if fib_levels else None,
            "volume_nodes": [{"price": v.price, "volume": v.volume, "is_hvn": v.is_hvn}
                           for v in (volume_nodes or [])],
            "support_levels": [l.to_dict() for l in clustered_levels if l.is_support],
            "resistance_levels": [l.to_dict() for l in clustered_levels if not l.is_support],
        }

    def _detect_swing_levels(
        self,
        candles: List[Dict],
        timeframe: str,
    ) -> List[KeyLevel]:
        """Detect swing highs and lows from candle data.

        A swing high is a high that is higher than the N candles before and after.
        A swing low is a low that is lower than the N candles before and after.
        """
        levels = []
        n = self.swing_lookback

        highs = [float(c.get("high", 0)) for c in candles]
        lows = [float(c.get("low", 0)) for c in candles]

        # Detect swing highs
        for i in range(n, len(highs) - n):
            is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
                           all(highs[i] >= highs[i+j] for j in range(1, n+1))

            if is_swing_high:
                levels.append(KeyLevel(
                    price=highs[i],
                    level_type=LevelType.SWING_HIGH,
                    is_support=False,  # Swing high is resistance
                    strength=LevelStrength.WEAK,
                    timeframe=timeframe,
                    test_count=1,
                ))

        # Detect swing lows
        for i in range(n, len(lows) - n):
            is_swing_low = all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
                          all(lows[i] <= lows[i+j] for j in range(1, n+1))

            if is_swing_low:
                levels.append(KeyLevel(
                    price=lows[i],
                    level_type=LevelType.SWING_LOW,
                    is_support=True,  # Swing low is support
                    strength=LevelStrength.WEAK,
                    timeframe=timeframe,
                    test_count=1,
                ))

        return levels

    def _calculate_fibonacci_levels(
        self,
        candles: List[Dict],
        lookback: int = 50,
    ) -> Optional[FibonacciLevels]:
        """Calculate Fibonacci retracement and extension levels.

        Uses the highest high and lowest low in the lookback period.
        Direction is determined by which came first.
        """
        if len(candles) < lookback:
            lookback = len(candles)

        recent_candles = candles[-lookback:]

        highs = [float(c.get("high", 0)) for c in recent_candles]
        lows = [float(c.get("low", 0)) for c in recent_candles]

        swing_high = max(highs)
        swing_low = min(lows)

        if swing_high <= swing_low:
            return None

        # Find which came first to determine direction
        high_idx = highs.index(swing_high)
        low_idx = lows.index(swing_low)

        # If low came before high, we're in an uptrend (retracement from high)
        # If high came before low, we're in a downtrend (retracement from low)
        direction = "up" if low_idx < high_idx else "down"

        return FibonacciLevels(
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
        )

    def _fib_to_key_levels(
        self,
        fib: FibonacciLevels,
        timeframe: str,
    ) -> List[KeyLevel]:
        """Convert Fibonacci levels to KeyLevel objects."""
        levels = []

        fib_dict = fib.get_all_levels()
        current_price = (fib.swing_high + fib.swing_low) / 2

        for name, price in fib_dict.items():
            if name in ["swing_high", "swing_low"]:
                continue

            is_support = price < current_price

            levels.append(KeyLevel(
                price=price,
                level_type=LevelType.FIBONACCI,
                is_support=is_support,
                strength=LevelStrength.MODERATE if "61.8" in name or "50.0" in name else LevelStrength.WEAK,
                timeframe=timeframe,
            ))

        return levels

    def _calculate_volume_profile(
        self,
        candles: List[Dict],
    ) -> List[VolumeNode]:
        """Calculate Volume Profile and identify High/Low Volume Nodes.

        High Volume Nodes (HVN): Areas where price spent a lot of time (S/R)
        Low Volume Nodes (LVN): Areas where price moved quickly through
        """
        if not candles:
            return []

        # Get price range
        all_highs = [float(c.get("high", 0)) for c in candles]
        all_lows = [float(c.get("low", 0)) for c in candles]

        price_high = max(all_highs)
        price_low = min(all_lows)

        if price_high <= price_low:
            return []

        # Create price bins
        bin_size = (price_high - price_low) / self.volume_profile_bins
        bins = np.zeros(self.volume_profile_bins)

        # Distribute volume across bins
        for candle in candles:
            c_high = float(candle.get("high", 0))
            c_low = float(candle.get("low", 0))
            c_volume = float(candle.get("volume", 0))

            if c_high <= c_low or c_volume <= 0:
                continue

            # Find which bins this candle covers
            low_bin = int((c_low - price_low) / bin_size)
            high_bin = int((c_high - price_low) / bin_size)

            low_bin = max(0, min(low_bin, self.volume_profile_bins - 1))
            high_bin = max(0, min(high_bin, self.volume_profile_bins - 1))

            # Distribute volume evenly across covered bins
            num_bins = high_bin - low_bin + 1
            vol_per_bin = c_volume / num_bins

            for b in range(low_bin, high_bin + 1):
                bins[b] += vol_per_bin

        # Calculate volume statistics
        mean_vol = np.mean(bins)
        std_vol = np.std(bins)

        # Identify High and Low Volume Nodes
        nodes = []
        hvn_threshold = mean_vol + std_vol
        lvn_threshold = mean_vol - std_vol * 0.5

        for i, vol in enumerate(bins):
            bin_low = price_low + i * bin_size
            bin_high = bin_low + bin_size
            bin_mid = (bin_low + bin_high) / 2

            is_hvn = vol >= hvn_threshold
            is_lvn = vol <= lvn_threshold and vol > 0

            if is_hvn or is_lvn:
                nodes.append(VolumeNode(
                    price=bin_mid,
                    volume=vol,
                    is_hvn=is_hvn,
                    is_lvn=is_lvn,
                    price_range=(bin_low, bin_high),
                ))

        return nodes

    def _volume_nodes_to_key_levels(
        self,
        nodes: List[VolumeNode],
        timeframe: str,
    ) -> List[KeyLevel]:
        """Convert Volume Nodes to KeyLevel objects (only HVN)."""
        levels = []

        for node in nodes:
            if not node.is_hvn:
                continue

            # HVN can act as both support and resistance
            # We create one level and it will be updated based on price action
            levels.append(KeyLevel(
                price=node.price,
                level_type=LevelType.VOLUME_NODE,
                is_support=True,  # Will be updated by clustering/testing
                strength=LevelStrength.MODERATE,
                timeframe=timeframe,
            ))

        return levels

    def _cluster_levels(
        self,
        levels: List[KeyLevel],
    ) -> List[KeyLevel]:
        """Cluster nearby levels into stronger combined levels.

        When multiple levels from different timeframes/types are close together,
        they form a stronger combined level.
        """
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda l: l.price)

        clusters: List[List[KeyLevel]] = []
        current_cluster: List[KeyLevel] = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            # Check if this level is close to the current cluster
            cluster_price = np.mean([l.price for l in current_cluster])
            distance = abs(level.price - cluster_price) / cluster_price

            if distance <= self.cluster_threshold:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]

        clusters.append(current_cluster)

        # Create combined levels from clusters
        combined_levels: List[KeyLevel] = []

        for cluster in clusters:
            if len(cluster) == 1:
                combined_levels.append(cluster[0])
            else:
                # Create a combined level
                avg_price = np.mean([l.price for l in cluster])
                total_tests = sum(l.test_count for l in cluster)

                # Determine if support or resistance based on majority
                support_count = sum(1 for l in cluster if l.is_support)
                is_support = support_count > len(cluster) / 2

                # Strength based on cluster size and timeframes
                timeframes = set(l.timeframe for l in cluster)
                if len(cluster) >= 4 or len(timeframes) >= 3:
                    strength = LevelStrength.VERY_STRONG
                elif len(cluster) >= 3 or len(timeframes) >= 2:
                    strength = LevelStrength.STRONG
                else:
                    strength = LevelStrength.MODERATE

                combined_levels.append(KeyLevel(
                    price=avg_price,
                    level_type=LevelType.CLUSTER,
                    is_support=is_support,
                    strength=strength,
                    timeframe=",".join(sorted(timeframes)),
                    test_count=total_tests,
                ))

        return combined_levels

    def _update_level_strength(
        self,
        levels: List[KeyLevel],
        current_price: float,
    ) -> None:
        """Update level strength based on current price position."""
        for level in levels:
            # Determine if level is support or resistance based on current price
            level.is_support = level.price < current_price

    async def _persist_levels(
        self,
        symbol: str,
        levels: List[KeyLevel],
    ) -> None:
        """Persist levels to Redis."""
        if not self.redis_client:
            return

        try:
            key = f"key_levels:{symbol}"
            data = [l.to_dict() for l in levels]
            await self.redis_client.set(key, data, ttl=86400)  # 24 hour TTL
        except Exception as e:
            logger.warning(f"Failed to persist key levels: {e}")

    async def get_nearest_levels(
        self,
        symbol: str,
        current_price: float,
        count: int = 3,
    ) -> Dict[str, List[Dict]]:
        """Get nearest support and resistance levels to current price.

        Args:
            symbol: Trading symbol
            current_price: Current price
            count: Number of levels to return on each side

        Returns:
            Dictionary with nearest support and resistance levels
        """
        levels = self._levels_cache.get(symbol, [])

        if not levels:
            return {"supports": [], "resistances": []}

        supports = [l for l in levels if l.price < current_price]
        resistances = [l for l in levels if l.price >= current_price]

        # Sort supports descending (nearest first)
        supports.sort(key=lambda l: l.price, reverse=True)
        # Sort resistances ascending (nearest first)
        resistances.sort(key=lambda l: l.price)

        return {
            "supports": [l.to_dict() for l in supports[:count]],
            "resistances": [l.to_dict() for l in resistances[:count]],
        }

    def get_fibonacci_levels(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get cached Fibonacci levels for a symbol."""
        fib = self._fib_cache.get(symbol)
        return fib.get_all_levels() if fib else None

    def get_volume_nodes(self, symbol: str) -> List[Dict]:
        """Get cached volume nodes for a symbol."""
        nodes = self._volume_nodes_cache.get(symbol, [])
        return [{"price": n.price, "volume": n.volume, "is_hvn": n.is_hvn, "is_lvn": n.is_lvn}
                for n in nodes]

    async def update_level_test(
        self,
        symbol: str,
        price: float,
        was_bounce: bool,
    ) -> None:
        """Update level statistics when price tests a level.

        Args:
            symbol: Trading symbol
            price: Price that tested the level
            was_bounce: True if price bounced, False if it broke through
        """
        levels = self._levels_cache.get(symbol, [])

        for level in levels:
            # Check if this price tested this level (within threshold)
            distance = abs(price - level.price) / level.price
            if distance <= self.cluster_threshold:
                level.test_count += 1
                level.last_test = datetime.utcnow()

                if was_bounce:
                    level.bounce_count += 1
                else:
                    level.break_count += 1

                # Update strength based on test count and bounce rate
                if level.test_count >= 7 and level.bounce_rate >= 0.6:
                    level.strength = LevelStrength.VERY_STRONG
                elif level.test_count >= 5:
                    level.strength = LevelStrength.STRONG
                elif level.test_count >= 3:
                    level.strength = LevelStrength.MODERATE

                break

        # Persist updated levels
        if self.redis_client:
            await self._persist_levels(symbol, levels)

    def get_level_summary(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Get a summary of key levels for a symbol.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            Summary dictionary
        """
        levels = self._levels_cache.get(symbol, [])
        fib = self._fib_cache.get(symbol)

        if not levels:
            return {
                "has_levels": False,
                "nearest_support": None,
                "nearest_resistance": None,
                "distance_to_support_pct": None,
                "distance_to_resistance_pct": None,
            }

        supports = sorted([l for l in levels if l.price < current_price],
                         key=lambda l: l.price, reverse=True)
        resistances = sorted([l for l in levels if l.price >= current_price],
                            key=lambda l: l.price)

        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None

        return {
            "has_levels": True,
            "total_levels": len(levels),
            "support_count": len(supports),
            "resistance_count": len(resistances),
            "nearest_support": nearest_support.to_dict() if nearest_support else None,
            "nearest_resistance": nearest_resistance.to_dict() if nearest_resistance else None,
            "distance_to_support_pct": (
                (current_price - nearest_support.price) / current_price * 100
            ) if nearest_support else None,
            "distance_to_resistance_pct": (
                (nearest_resistance.price - current_price) / current_price * 100
            ) if nearest_resistance else None,
            "fibonacci": fib.get_all_levels() if fib else None,
            "strong_levels": [l.to_dict() for l in levels
                            if l.strength in [LevelStrength.STRONG, LevelStrength.VERY_STRONG]],
        }
