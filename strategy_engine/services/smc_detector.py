"""SMC (Smart Money Concepts) Detector Service.

Comprehensive Smart Money Concepts detection including:
1. Order Blocks (OB) - Last opposing candle before impulse move
2. Fair Value Gaps (FVG) - Price imbalances/inefficiencies
3. Break of Structure (BOS) - Trend continuation signals
4. Change of Character (CHoCH) - Trend reversal signals
5. Liquidity Pools - Stop loss clusters (equal highs/lows)
6. Premium/Discount Zones - Value areas based on Fibonacci 50%
7. Inducement - False breakouts designed to trap traders
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import json
from loguru import logger

if TYPE_CHECKING:
    from .market_data_service import MarketDataService
    from .redis_client import RedisClient
    from .key_level_detector import KeyLevelDetector


class SMCType(Enum):
    """Type of SMC concept."""
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    BREAK_OF_STRUCTURE = "break_of_structure"
    CHANGE_OF_CHARACTER = "change_of_character"
    LIQUIDITY_POOL = "liquidity_pool"
    INDUCEMENT = "inducement"


class SMCDirection(Enum):
    """Direction of SMC signal."""
    BULLISH = "bullish"
    BEARISH = "bearish"


class SMCStrength(Enum):
    """Strength of SMC concept."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class MarketStructure(Enum):
    """Overall market structure."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


@dataclass
class OrderBlock:
    """An Order Block - last opposing candle before an impulsive move.

    Order Blocks represent institutional entry zones where smart money
    placed significant orders. They act as strong S/R zones.
    """
    direction: SMCDirection
    price_high: float
    price_low: float
    midpoint: float
    impulse_size_pct: float  # Size of the impulse move that followed
    timeframe: str
    timestamp: datetime
    tested: bool = False  # Has price returned to this zone?
    mitigated: bool = False  # Has price passed through this zone?
    test_count: int = 0
    strength: SMCStrength = SMCStrength.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "price_high": float(self.price_high),
            "price_low": float(self.price_low),
            "midpoint": float(self.midpoint),
            "impulse_size_pct": float(self.impulse_size_pct),
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tested": self.tested,
            "mitigated": self.mitigated,
            "test_count": self.test_count,
            "strength": self.strength.value,
        }


@dataclass
class FairValueGap:
    """A Fair Value Gap - price imbalance/inefficiency.

    FVGs occur when price moves so quickly that a gap is left between
    candles. Price tends to return to fill these gaps.
    """
    direction: SMCDirection
    gap_high: float  # Upper boundary of the gap
    gap_low: float  # Lower boundary of the gap
    midpoint: float
    gap_size_pct: float  # Size of gap as percentage
    timeframe: str
    timestamp: datetime
    fill_pct: float = 0.0  # How much of the gap has been filled
    filled: bool = False  # Has gap been fully filled?
    strength: SMCStrength = SMCStrength.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "gap_high": float(self.gap_high),
            "gap_low": float(self.gap_low),
            "midpoint": float(self.midpoint),
            "gap_size_pct": float(self.gap_size_pct),
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "fill_pct": float(self.fill_pct),
            "filled": self.filled,
            "strength": self.strength.value,
        }


@dataclass
class BreakOfStructure:
    """Break of Structure (BOS) - Trend continuation signal.

    BOS occurs when price breaks a previous swing high/low in the
    direction of the current trend, confirming continuation.
    """
    direction: SMCDirection
    break_price: float  # Price level that was broken
    break_candle_idx: int  # Index of candle that caused the break
    timeframe: str
    timestamp: datetime
    confirmed: bool = True  # Was break confirmed by close?
    retest_level: Optional[float] = None  # Price to watch for retest

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "break_price": float(self.break_price),
            "break_candle_idx": self.break_candle_idx,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confirmed": self.confirmed,
            "retest_level": float(self.retest_level) if self.retest_level else None,
        }


@dataclass
class ChangeOfCharacter:
    """Change of Character (CHoCH) - Trend reversal signal.

    CHoCH is the first break of structure against the prevailing trend,
    signaling a potential trend reversal. It's an early reversal signal.
    """
    direction: SMCDirection  # Direction of the NEW trend
    choch_price: float  # Price level where CHoCH occurred
    old_trend: MarketStructure  # The previous trend
    break_candle_idx: int
    timeframe: str
    timestamp: datetime
    confirmed: bool = False  # Needs follow-through to confirm

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "choch_price": float(self.choch_price),
            "old_trend": self.old_trend.value,
            "break_candle_idx": self.break_candle_idx,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confirmed": self.confirmed,
        }


@dataclass
class LiquidityPool:
    """Liquidity Pool - Cluster of stop losses.

    Liquidity pools form at equal highs/lows where stops accumulate.
    Smart money often hunts these pools before reversing.
    """
    side: str  # "buy" (above price, shorts' stops) or "sell" (below price, longs' stops)
    price_level: float
    source: str  # "equal_highs", "equal_lows", "swing_high", "swing_low"
    num_touches: int  # How many times price tested this level
    timeframe: str
    timestamp: datetime
    swept: bool = False  # Has this liquidity been taken?
    strength: SMCStrength = SMCStrength.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "side": self.side,
            "price_level": float(self.price_level),
            "source": self.source,
            "num_touches": self.num_touches,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "swept": self.swept,
            "strength": self.strength.value,
        }


@dataclass
class PremiumDiscountZone:
    """Premium/Discount Zone based on Fibonacci 50%.

    - Premium Zone (above 61.8%): Sell zone, price is expensive
    - Discount Zone (below 38.2%): Buy zone, price is cheap
    - Equilibrium (around 50%): Fair value
    """
    swing_high: float
    swing_low: float
    equilibrium: float  # 50% level
    premium_start: float  # 61.8% level
    discount_end: float  # 38.2% level
    current_zone: str  # "premium", "discount", or "equilibrium"
    distance_pct: float  # Distance from equilibrium as percentage
    timeframe: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "swing_high": float(self.swing_high),
            "swing_low": float(self.swing_low),
            "equilibrium": float(self.equilibrium),
            "premium_start": float(self.premium_start),
            "discount_end": float(self.discount_end),
            "current_zone": self.current_zone,
            "distance_pct": float(self.distance_pct),
            "timeframe": self.timeframe,
        }


@dataclass
class Inducement:
    """Inducement - False breakout designed to trap traders.

    Inducements are engineered moves that trigger stops before the
    real move happens. They're often quick wicks or failed breakouts.
    """
    direction: SMCDirection  # Direction of the trap (opposite of real move)
    inducement_level: float  # Level that was swept
    sweep_high: float  # Highest point of the sweep
    sweep_low: float  # Lowest point of the sweep
    timeframe: str
    timestamp: datetime
    reversal_confirmed: bool = False  # Did price reverse after the sweep?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "inducement_level": float(self.inducement_level),
            "sweep_high": float(self.sweep_high),
            "sweep_low": float(self.sweep_low),
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "reversal_confirmed": self.reversal_confirmed,
        }


@dataclass
class MarketStructureState:
    """Current market structure state."""
    trend: MarketStructure
    swing_highs: List[Tuple[int, float]]  # (index, price) of swing highs
    swing_lows: List[Tuple[int, float]]  # (index, price) of swing lows
    last_bos: Optional[BreakOfStructure] = None
    last_choch: Optional[ChangeOfCharacter] = None
    hh_count: int = 0  # Higher high count
    hl_count: int = 0  # Higher low count
    lh_count: int = 0  # Lower high count
    ll_count: int = 0  # Lower low count
    timeframe: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trend": self.trend.value,
            "swing_highs": [(int(i), float(p)) for i, p in self.swing_highs[-5:]],
            "swing_lows": [(int(i), float(p)) for i, p in self.swing_lows[-5:]],
            "last_bos": self.last_bos.to_dict() if self.last_bos else None,
            "last_choch": self.last_choch.to_dict() if self.last_choch else None,
            "hh_count": self.hh_count,
            "hl_count": self.hl_count,
            "lh_count": self.lh_count,
            "ll_count": self.ll_count,
            "timeframe": self.timeframe,
        }


class SMCDetector:
    """Smart Money Concepts Detector.

    Provides comprehensive SMC analysis including:
    - Order Block detection
    - Fair Value Gap detection
    - Break of Structure / Change of Character
    - Liquidity Pool identification
    - Premium/Discount zone calculation
    - Inducement detection
    """

    def __init__(
        self,
        market_data_service: Optional["MarketDataService"] = None,
        redis_client: Optional["RedisClient"] = None,
        key_level_detector: Optional["KeyLevelDetector"] = None,
        swing_lookback: int = 5,
        ob_lookback: int = 20,
        fvg_lookback: int = 50,
        min_impulse_pct: float = 0.015,  # 1.5% minimum impulse for OB
        min_fvg_pct: float = 0.003,  # 0.3% minimum gap size
        equal_level_tolerance: float = 0.002,  # 0.2% for equal highs/lows
    ):
        """Initialize SMCDetector.

        Args:
            market_data_service: For fetching candle data
            redis_client: For persisting SMC data
            key_level_detector: For integration with S/R levels
            swing_lookback: Candles to look back for swing detection
            ob_lookback: Candles to look back for order block detection
            fvg_lookback: Candles to look back for FVG detection
            min_impulse_pct: Minimum impulse size for order blocks (1.5%)
            min_fvg_pct: Minimum gap size for FVGs (0.3%)
            equal_level_tolerance: Tolerance for equal highs/lows (0.2%)
        """
        self.market_data_service = market_data_service
        self.redis_client = redis_client
        self.key_level_detector = key_level_detector
        self.swing_lookback = swing_lookback
        self.ob_lookback = ob_lookback
        self.fvg_lookback = fvg_lookback
        self.min_impulse_pct = min_impulse_pct
        self.min_fvg_pct = min_fvg_pct
        self.equal_level_tolerance = equal_level_tolerance

        # Cache for SMC data by symbol
        self._order_blocks_cache: Dict[str, List[OrderBlock]] = {}
        self._fvg_cache: Dict[str, List[FairValueGap]] = {}
        self._structure_cache: Dict[str, MarketStructureState] = {}
        self._liquidity_pools_cache: Dict[str, List[LiquidityPool]] = {}
        self._premium_discount_cache: Dict[str, PremiumDiscountZone] = {}
        self._inducements_cache: Dict[str, List[Inducement]] = {}

    async def detect_all(
        self,
        symbol: str,
        candles_1h: List[Dict] = None,
        candles_4h: List[Dict] = None,
        candles_1d: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Detect all SMC concepts for a symbol.

        Args:
            symbol: Trading symbol
            candles_1h: 1-hour candle data
            candles_4h: 4-hour candle data
            candles_1d: Daily candle data

        Returns:
            Dictionary with all detected SMC concepts
        """
        result = {
            "order_blocks": [],
            "fair_value_gaps": [],
            "market_structure": {},
            "liquidity_pools": [],
            "premium_discount": {},
            "inducements": [],
            "bos_events": [],
            "choch_events": [],
        }

        # Process each timeframe
        for candles, timeframe in [
            (candles_1h, "1h"),
            (candles_4h, "4h"),
            (candles_1d, "1d"),
        ]:
            if not candles or len(candles) < self.swing_lookback * 2:
                continue

            try:
                # 1. Detect swing structure (foundation for everything else)
                swings = self._detect_swing_structure(candles, timeframe)

                # 2. Analyze market structure (trend, HH/HL/LH/LL)
                structure = self._analyze_market_structure(swings, candles, timeframe)

                # Store the highest timeframe structure
                if timeframe == "1d" or (timeframe == "4h" and "1d" not in result["market_structure"]):
                    result["market_structure"] = structure.to_dict()
                    self._structure_cache[symbol] = structure

                # 3. Detect Order Blocks
                order_blocks = self._detect_order_blocks(candles, swings, timeframe)
                result["order_blocks"].extend([ob.to_dict() for ob in order_blocks])

                # 4. Detect Fair Value Gaps
                fvgs = self._detect_fair_value_gaps(candles, timeframe)
                result["fair_value_gaps"].extend([fvg.to_dict() for fvg in fvgs])

                # 5. Detect BOS and CHoCH
                bos_choch = self._detect_structure_breaks(candles, swings, structure, timeframe)
                result["bos_events"].extend([b.to_dict() for b in bos_choch.get("bos", [])])
                result["choch_events"].extend([c.to_dict() for c in bos_choch.get("choch", [])])

                # 6. Detect Liquidity Pools
                pools = self._detect_liquidity_pools(candles, swings, timeframe)
                result["liquidity_pools"].extend([p.to_dict() for p in pools])

                # 7. Calculate Premium/Discount Zones (use 4h or 1d)
                if timeframe in ["4h", "1d"]:
                    prem_disc = self._calculate_premium_discount(candles, swings, timeframe)
                    if prem_disc:
                        result["premium_discount"] = prem_disc.to_dict()
                        self._premium_discount_cache[symbol] = prem_disc

                # 8. Detect Inducements
                inducements = self._detect_inducements(candles, swings, pools, timeframe)
                result["inducements"].extend([ind.to_dict() for ind in inducements])

            except Exception as e:
                logger.warning(f"SMC detection failed for {symbol} {timeframe}: {e}")
                continue

        # Update caches
        self._order_blocks_cache[symbol] = [
            ob for ob in self._parse_order_blocks(result["order_blocks"])
        ]
        self._fvg_cache[symbol] = [
            fvg for fvg in self._parse_fvgs(result["fair_value_gaps"])
        ]
        self._liquidity_pools_cache[symbol] = [
            pool for pool in self._parse_liquidity_pools(result["liquidity_pools"])
        ]

        # Persist to Redis
        await self._persist_smc_data(symbol, result)

        return result

    def _detect_swing_structure(
        self,
        candles: List[Dict],
        timeframe: str,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Detect swing highs and lows.

        Args:
            candles: OHLCV candle data
            timeframe: Timeframe string

        Returns:
            Dictionary with swing_highs and swing_lows lists
        """
        swing_highs = []
        swing_lows = []
        n = self.swing_lookback

        highs = [float(c.get("high", 0)) for c in candles]
        lows = [float(c.get("low", 0)) for c in candles]

        for i in range(n, len(highs) - n):
            # Check swing high
            is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
                           all(highs[i] >= highs[i+j] for j in range(1, n+1))
            if is_swing_high:
                swing_highs.append((i, highs[i]))

            # Check swing low
            is_swing_low = all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
                          all(lows[i] <= lows[i+j] for j in range(1, n+1))
            if is_swing_low:
                swing_lows.append((i, lows[i]))

        return {
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
        }

    def _analyze_market_structure(
        self,
        swings: Dict[str, List[Tuple[int, float]]],
        candles: List[Dict],
        timeframe: str,
    ) -> MarketStructureState:
        """Analyze market structure from swings.

        Counts HH/HL (bullish) vs LH/LL (bearish) to determine trend.

        Args:
            swings: Dictionary with swing_highs and swing_lows
            candles: OHLCV data
            timeframe: Timeframe string

        Returns:
            MarketStructureState with trend analysis
        """
        swing_highs = swings.get("swing_highs", [])
        swing_lows = swings.get("swing_lows", [])

        hh_count = 0
        hl_count = 0
        lh_count = 0
        ll_count = 0

        # Compare consecutive swing highs
        for i in range(1, len(swing_highs)):
            if swing_highs[i][1] > swing_highs[i-1][1]:
                hh_count += 1
            else:
                lh_count += 1

        # Compare consecutive swing lows
        for i in range(1, len(swing_lows)):
            if swing_lows[i][1] > swing_lows[i-1][1]:
                hl_count += 1
            else:
                ll_count += 1

        # Determine trend
        bullish_score = hh_count + hl_count
        bearish_score = lh_count + ll_count

        if bullish_score > bearish_score + 1:
            trend = MarketStructure.BULLISH
        elif bearish_score > bullish_score + 1:
            trend = MarketStructure.BEARISH
        else:
            trend = MarketStructure.RANGING

        return MarketStructureState(
            trend=trend,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            hh_count=hh_count,
            hl_count=hl_count,
            lh_count=lh_count,
            ll_count=ll_count,
            timeframe=timeframe,
        )

    def _detect_order_blocks(
        self,
        candles: List[Dict],
        swings: Dict[str, List[Tuple[int, float]]],
        timeframe: str,
    ) -> List[OrderBlock]:
        """Detect Order Blocks.

        An Order Block is the last opposing candle before an impulsive move.

        Bullish OB: Last red candle before a strong up move
        Bearish OB: Last green candle before a strong down move

        Args:
            candles: OHLCV data
            swings: Swing highs/lows
            timeframe: Timeframe string

        Returns:
            List of OrderBlock objects
        """
        order_blocks = []
        lookback = min(self.ob_lookback, len(candles) - 1)

        for i in range(lookback, len(candles) - 1):
            open_price = float(candles[i].get("open", 0))
            close_price = float(candles[i].get("close", 0))
            high_price = float(candles[i].get("high", 0))
            low_price = float(candles[i].get("low", 0))

            if open_price == 0 or close_price == 0:
                continue

            # Check for bullish impulse after this candle
            if i + 3 < len(candles):
                # Measure the move over next 3 candles
                future_high = max(float(candles[i+j].get("high", 0)) for j in range(1, 4))
                impulse_up = (future_high - close_price) / close_price if close_price > 0 else 0

                # Bullish OB: Last bearish candle before bullish impulse
                if impulse_up >= self.min_impulse_pct and close_price < open_price:
                    timestamp = candles[i].get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp / 1000)
                    elif isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.utcnow()

                    order_blocks.append(OrderBlock(
                        direction=SMCDirection.BULLISH,
                        price_high=high_price,
                        price_low=low_price,
                        midpoint=(high_price + low_price) / 2,
                        impulse_size_pct=impulse_up * 100,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        strength=self._calculate_ob_strength(impulse_up),
                    ))

            # Check for bearish impulse after this candle
            if i + 3 < len(candles):
                future_low = min(float(candles[i+j].get("low", 0)) for j in range(1, 4))
                impulse_down = (close_price - future_low) / close_price if close_price > 0 else 0

                # Bearish OB: Last bullish candle before bearish impulse
                if impulse_down >= self.min_impulse_pct and close_price > open_price:
                    timestamp = candles[i].get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp / 1000)
                    elif isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.utcnow()

                    order_blocks.append(OrderBlock(
                        direction=SMCDirection.BEARISH,
                        price_high=high_price,
                        price_low=low_price,
                        midpoint=(high_price + low_price) / 2,
                        impulse_size_pct=impulse_down * 100,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        strength=self._calculate_ob_strength(impulse_down),
                    ))

        # Limit to most recent order blocks
        return order_blocks[-10:]

    def _calculate_ob_strength(self, impulse_pct: float) -> SMCStrength:
        """Calculate order block strength based on impulse size."""
        if impulse_pct >= 0.05:  # 5%+
            return SMCStrength.VERY_STRONG
        elif impulse_pct >= 0.03:  # 3%+
            return SMCStrength.STRONG
        elif impulse_pct >= 0.02:  # 2%+
            return SMCStrength.MODERATE
        else:
            return SMCStrength.WEAK

    def _detect_fair_value_gaps(
        self,
        candles: List[Dict],
        timeframe: str,
    ) -> List[FairValueGap]:
        """Detect Fair Value Gaps (FVGs).

        FVG pattern (bullish):
        - Candle 1's high < Candle 3's low
        - Gap between candle 1 high and candle 3 low

        FVG pattern (bearish):
        - Candle 1's low > Candle 3's high
        - Gap between candle 1 low and candle 3 high

        Args:
            candles: OHLCV data
            timeframe: Timeframe string

        Returns:
            List of FairValueGap objects
        """
        fvgs = []
        lookback = min(self.fvg_lookback, len(candles))

        for i in range(2, lookback):
            c1 = candles[i - 2]
            c3 = candles[i]

            c1_high = float(c1.get("high", 0))
            c1_low = float(c1.get("low", 0))
            c3_high = float(c3.get("high", 0))
            c3_low = float(c3.get("low", 0))

            if c1_high == 0 or c3_low == 0:
                continue

            # Bullish FVG: gap between c1 high and c3 low
            if c1_high < c3_low:
                gap_size = c3_low - c1_high
                gap_pct = gap_size / c1_high if c1_high > 0 else 0

                if gap_pct >= self.min_fvg_pct:
                    timestamp = c3.get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp / 1000)
                    elif isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.utcnow()

                    fvgs.append(FairValueGap(
                        direction=SMCDirection.BULLISH,
                        gap_high=c3_low,
                        gap_low=c1_high,
                        midpoint=(c3_low + c1_high) / 2,
                        gap_size_pct=gap_pct * 100,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        strength=self._calculate_fvg_strength(gap_pct),
                    ))

            # Bearish FVG: gap between c3 high and c1 low
            if c1_low > c3_high:
                gap_size = c1_low - c3_high
                gap_pct = gap_size / c1_low if c1_low > 0 else 0

                if gap_pct >= self.min_fvg_pct:
                    timestamp = c3.get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp / 1000)
                    elif isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.utcnow()

                    fvgs.append(FairValueGap(
                        direction=SMCDirection.BEARISH,
                        gap_high=c1_low,
                        gap_low=c3_high,
                        midpoint=(c1_low + c3_high) / 2,
                        gap_size_pct=gap_pct * 100,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        strength=self._calculate_fvg_strength(gap_pct),
                    ))

        return fvgs[-15:]  # Keep most recent

    def _calculate_fvg_strength(self, gap_pct: float) -> SMCStrength:
        """Calculate FVG strength based on gap size."""
        if gap_pct >= 0.02:  # 2%+
            return SMCStrength.VERY_STRONG
        elif gap_pct >= 0.01:  # 1%+
            return SMCStrength.STRONG
        elif gap_pct >= 0.005:  # 0.5%+
            return SMCStrength.MODERATE
        else:
            return SMCStrength.WEAK

    def _detect_structure_breaks(
        self,
        candles: List[Dict],
        swings: Dict[str, List[Tuple[int, float]]],
        structure: MarketStructureState,
        timeframe: str,
    ) -> Dict[str, List]:
        """Detect Break of Structure (BOS) and Change of Character (CHoCH).

        BOS: Break in the direction of the trend (continuation)
        CHoCH: First break against the trend (reversal signal)

        Args:
            candles: OHLCV data
            swings: Swing highs/lows
            structure: Current market structure
            timeframe: Timeframe string

        Returns:
            Dictionary with 'bos' and 'choch' lists
        """
        bos_events = []
        choch_events = []

        swing_highs = swings.get("swing_highs", [])
        swing_lows = swings.get("swing_lows", [])
        current_trend = structure.trend

        # Get recent price action
        if len(candles) < 5:
            return {"bos": bos_events, "choch": choch_events}

        recent_high = max(float(c.get("high", 0)) for c in candles[-5:])
        recent_low = min(float(c.get("low", 0)) for c in candles[-5:])
        current_close = float(candles[-1].get("close", 0))

        # Check for breaks of recent swing levels
        for i, (idx, level) in enumerate(swing_highs[-3:]):
            # Did we break above this swing high?
            if current_close > level:
                timestamp = candles[-1].get("timestamp")
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                if current_trend == MarketStructure.BULLISH:
                    # BOS - continuation of bullish trend
                    bos_events.append(BreakOfStructure(
                        direction=SMCDirection.BULLISH,
                        break_price=level,
                        break_candle_idx=len(candles) - 1,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        confirmed=True,
                        retest_level=level,
                    ))
                elif current_trend == MarketStructure.BEARISH:
                    # CHoCH - first bullish break in bearish trend
                    choch_events.append(ChangeOfCharacter(
                        direction=SMCDirection.BULLISH,
                        choch_price=level,
                        old_trend=MarketStructure.BEARISH,
                        break_candle_idx=len(candles) - 1,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        confirmed=False,
                    ))

        for i, (idx, level) in enumerate(swing_lows[-3:]):
            # Did we break below this swing low?
            if current_close < level:
                timestamp = candles[-1].get("timestamp")
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                if current_trend == MarketStructure.BEARISH:
                    # BOS - continuation of bearish trend
                    bos_events.append(BreakOfStructure(
                        direction=SMCDirection.BEARISH,
                        break_price=level,
                        break_candle_idx=len(candles) - 1,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        confirmed=True,
                        retest_level=level,
                    ))
                elif current_trend == MarketStructure.BULLISH:
                    # CHoCH - first bearish break in bullish trend
                    choch_events.append(ChangeOfCharacter(
                        direction=SMCDirection.BEARISH,
                        choch_price=level,
                        old_trend=MarketStructure.BULLISH,
                        break_candle_idx=len(candles) - 1,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        confirmed=False,
                    ))

        # Update structure with latest events
        if bos_events:
            structure.last_bos = bos_events[-1]
        if choch_events:
            structure.last_choch = choch_events[-1]

        return {"bos": bos_events, "choch": choch_events}

    def _detect_liquidity_pools(
        self,
        candles: List[Dict],
        swings: Dict[str, List[Tuple[int, float]]],
        timeframe: str,
    ) -> List[LiquidityPool]:
        """Detect Liquidity Pools.

        Liquidity pools form at:
        1. Equal highs (within tolerance) - buy-side liquidity (shorts' stops)
        2. Equal lows (within tolerance) - sell-side liquidity (longs' stops)
        3. Swing highs - buy-side liquidity
        4. Swing lows - sell-side liquidity

        Args:
            candles: OHLCV data
            swings: Swing highs/lows
            timeframe: Timeframe string

        Returns:
            List of LiquidityPool objects
        """
        pools = []
        swing_highs = swings.get("swing_highs", [])
        swing_lows = swings.get("swing_lows", [])

        # Find equal highs
        highs = [float(c.get("high", 0)) for c in candles]
        for i in range(len(highs)):
            touches = 1
            for j in range(i + 1, len(highs)):
                if abs(highs[i] - highs[j]) / highs[i] <= self.equal_level_tolerance:
                    touches += 1

            if touches >= 2:
                timestamp = candles[i].get("timestamp")
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                pools.append(LiquidityPool(
                    side="buy",  # Buy-side liquidity (above price)
                    price_level=highs[i],
                    source="equal_highs",
                    num_touches=touches,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    strength=self._calculate_pool_strength(touches),
                ))

        # Find equal lows
        lows = [float(c.get("low", 0)) for c in candles]
        for i in range(len(lows)):
            touches = 1
            for j in range(i + 1, len(lows)):
                if abs(lows[i] - lows[j]) / lows[i] <= self.equal_level_tolerance:
                    touches += 1

            if touches >= 2:
                timestamp = candles[i].get("timestamp")
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                pools.append(LiquidityPool(
                    side="sell",  # Sell-side liquidity (below price)
                    price_level=lows[i],
                    source="equal_lows",
                    num_touches=touches,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    strength=self._calculate_pool_strength(touches),
                ))

        # Add swing highs as buy-side liquidity
        for idx, level in swing_highs[-5:]:
            timestamp = candles[idx].get("timestamp") if idx < len(candles) else datetime.utcnow()
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            pools.append(LiquidityPool(
                side="buy",
                price_level=level,
                source="swing_high",
                num_touches=1,
                timeframe=timeframe,
                timestamp=timestamp,
                strength=SMCStrength.MODERATE,
            ))

        # Add swing lows as sell-side liquidity
        for idx, level in swing_lows[-5:]:
            timestamp = candles[idx].get("timestamp") if idx < len(candles) else datetime.utcnow()
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            pools.append(LiquidityPool(
                side="sell",
                price_level=level,
                source="swing_low",
                num_touches=1,
                timeframe=timeframe,
                timestamp=timestamp,
                strength=SMCStrength.MODERATE,
            ))

        return pools[-15:]

    def _calculate_pool_strength(self, touches: int) -> SMCStrength:
        """Calculate liquidity pool strength based on touches."""
        if touches >= 4:
            return SMCStrength.VERY_STRONG
        elif touches >= 3:
            return SMCStrength.STRONG
        elif touches >= 2:
            return SMCStrength.MODERATE
        else:
            return SMCStrength.WEAK

    def _calculate_premium_discount(
        self,
        candles: List[Dict],
        swings: Dict[str, List[Tuple[int, float]]],
        timeframe: str,
    ) -> Optional[PremiumDiscountZone]:
        """Calculate Premium/Discount zones.

        Based on Fibonacci 50% of the range:
        - Premium: Above 61.8% (expensive, sell zone)
        - Discount: Below 38.2% (cheap, buy zone)
        - Equilibrium: Around 50% (fair value)

        Args:
            candles: OHLCV data
            swings: Swing highs/lows
            timeframe: Timeframe string

        Returns:
            PremiumDiscountZone or None
        """
        swing_highs = swings.get("swing_highs", [])
        swing_lows = swings.get("swing_lows", [])

        if not swing_highs or not swing_lows:
            return None

        # Use most recent significant swing high and low
        swing_high = max(h[1] for h in swing_highs[-3:])
        swing_low = min(l[1] for l in swing_lows[-3:])

        if swing_high <= swing_low:
            return None

        range_size = swing_high - swing_low
        equilibrium = swing_low + (range_size * 0.5)
        premium_start = swing_low + (range_size * 0.618)
        discount_end = swing_low + (range_size * 0.382)

        # Get current price
        current_price = float(candles[-1].get("close", 0))
        if current_price <= 0:
            return None

        # Determine current zone
        if current_price >= premium_start:
            current_zone = "premium"
        elif current_price <= discount_end:
            current_zone = "discount"
        else:
            current_zone = "equilibrium"

        # Calculate distance from equilibrium
        distance_pct = ((current_price - equilibrium) / equilibrium) * 100

        return PremiumDiscountZone(
            swing_high=swing_high,
            swing_low=swing_low,
            equilibrium=equilibrium,
            premium_start=premium_start,
            discount_end=discount_end,
            current_zone=current_zone,
            distance_pct=distance_pct,
            timeframe=timeframe,
        )

    def _detect_inducements(
        self,
        candles: List[Dict],
        swings: Dict[str, List[Tuple[int, float]]],
        pools: List[LiquidityPool],
        timeframe: str,
    ) -> List[Inducement]:
        """Detect Inducements (false breakouts/stop hunts).

        An inducement is when price:
        1. Sweeps a liquidity level (goes past it)
        2. Then reverses sharply

        Args:
            candles: OHLCV data
            swings: Swing highs/lows
            pools: Detected liquidity pools
            timeframe: Timeframe string

        Returns:
            List of Inducement objects
        """
        inducements = []

        if len(candles) < 5:
            return inducements

        # Check recent candles for sweep patterns
        for i in range(len(candles) - 5, len(candles) - 1):
            if i < 0:
                continue

            candle = candles[i]
            next_candle = candles[i + 1]

            high = float(candle.get("high", 0))
            low = float(candle.get("low", 0))
            close = float(candle.get("close", 0))
            next_close = float(next_candle.get("close", 0))

            # Check for bullish sweep (dip below then reverse up)
            for pool in pools:
                if pool.side == "sell" and pool.price_level > 0:
                    # Did we sweep below this level?
                    if low < pool.price_level and close > pool.price_level:
                        # Did we reverse up?
                        if next_close > close:
                            timestamp = candle.get("timestamp")
                            if isinstance(timestamp, (int, float)):
                                timestamp = datetime.fromtimestamp(timestamp / 1000)
                            elif isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            else:
                                timestamp = datetime.utcnow()

                            inducements.append(Inducement(
                                direction=SMCDirection.BULLISH,
                                inducement_level=pool.price_level,
                                sweep_high=high,
                                sweep_low=low,
                                timeframe=timeframe,
                                timestamp=timestamp,
                                reversal_confirmed=True,
                            ))
                            pool.swept = True

                # Check for bearish sweep (spike above then reverse down)
                if pool.side == "buy" and pool.price_level > 0:
                    # Did we sweep above this level?
                    if high > pool.price_level and close < pool.price_level:
                        # Did we reverse down?
                        if next_close < close:
                            timestamp = candle.get("timestamp")
                            if isinstance(timestamp, (int, float)):
                                timestamp = datetime.fromtimestamp(timestamp / 1000)
                            elif isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            else:
                                timestamp = datetime.utcnow()

                            inducements.append(Inducement(
                                direction=SMCDirection.BEARISH,
                                inducement_level=pool.price_level,
                                sweep_high=high,
                                sweep_low=low,
                                timeframe=timeframe,
                                timestamp=timestamp,
                                reversal_confirmed=True,
                            ))
                            pool.swept = True

        return inducements[-5:]

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_nearest_order_block(
        self,
        symbol: str,
        current_price: float,
        direction: Optional[SMCDirection] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get nearest order block to current price.

        Args:
            symbol: Trading symbol
            current_price: Current price
            direction: Optional filter by direction (bullish/bearish)

        Returns:
            Nearest order block or None
        """
        order_blocks = self._order_blocks_cache.get(symbol, [])

        if not order_blocks:
            return None

        # Filter by direction if specified
        if direction:
            order_blocks = [ob for ob in order_blocks if ob.direction == direction]

        if not order_blocks:
            return None

        # Find nearest by midpoint distance
        nearest = min(
            order_blocks,
            key=lambda ob: abs(ob.midpoint - current_price)
        )

        return nearest.to_dict()

    async def get_active_fvgs(
        self,
        symbol: str,
        current_price: float,
        max_distance_pct: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Get active (unfilled) FVGs near current price.

        Args:
            symbol: Trading symbol
            current_price: Current price
            max_distance_pct: Maximum distance from price (default 5%)

        Returns:
            List of active FVG dictionaries
        """
        fvgs = self._fvg_cache.get(symbol, [])

        if not fvgs:
            return []

        active_fvgs = []
        for fvg in fvgs:
            if fvg.filled:
                continue

            # Calculate distance from current price
            distance = abs(fvg.midpoint - current_price) / current_price * 100

            if distance <= max_distance_pct:
                fvg_dict = fvg.to_dict()
                fvg_dict["distance_pct"] = distance
                active_fvgs.append(fvg_dict)

        # Sort by distance
        active_fvgs.sort(key=lambda f: f["distance_pct"])

        return active_fvgs[:10]

    def get_smc_summary(
        self,
        symbol: str,
        current_price: float,
    ) -> Dict[str, Any]:
        """Get a summary of SMC analysis for a symbol.

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            Summary dictionary
        """
        structure = self._structure_cache.get(symbol)
        prem_disc = self._premium_discount_cache.get(symbol)
        order_blocks = self._order_blocks_cache.get(symbol, [])
        fvgs = self._fvg_cache.get(symbol, [])
        pools = self._liquidity_pools_cache.get(symbol, [])

        # Find nearest bullish and bearish OBs
        bullish_obs = [ob for ob in order_blocks if ob.direction == SMCDirection.BULLISH]
        bearish_obs = [ob for ob in order_blocks if ob.direction == SMCDirection.BEARISH]

        nearest_bullish_ob = None
        if bullish_obs:
            below_price = [ob for ob in bullish_obs if ob.midpoint < current_price]
            if below_price:
                nearest_bullish_ob = max(below_price, key=lambda ob: ob.midpoint).to_dict()

        nearest_bearish_ob = None
        if bearish_obs:
            above_price = [ob for ob in bearish_obs if ob.midpoint > current_price]
            if above_price:
                nearest_bearish_ob = min(above_price, key=lambda ob: ob.midpoint).to_dict()

        # Count active FVGs
        active_fvgs = [fvg for fvg in fvgs if not fvg.filled]

        return {
            "trend": structure.trend.value if structure else "unknown",
            "premium_discount": prem_disc.to_dict() if prem_disc else None,
            "nearest_bullish_ob": nearest_bullish_ob,
            "nearest_bearish_ob": nearest_bearish_ob,
            "active_ob_count": len([ob for ob in order_blocks if not ob.mitigated]),
            "active_fvg_count": len(active_fvgs),
            "buy_liquidity_pools": len([p for p in pools if p.side == "buy" and not p.swept]),
            "sell_liquidity_pools": len([p for p in pools if p.side == "sell" and not p.swept]),
            "structure_summary": {
                "hh_count": structure.hh_count if structure else 0,
                "hl_count": structure.hl_count if structure else 0,
                "lh_count": structure.lh_count if structure else 0,
                "ll_count": structure.ll_count if structure else 0,
                "last_bos": structure.last_bos.to_dict() if structure and structure.last_bos else None,
                "last_choch": structure.last_choch.to_dict() if structure and structure.last_choch else None,
            },
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    async def _persist_smc_data(
        self,
        symbol: str,
        data: Dict[str, Any],
    ) -> None:
        """Persist SMC data to Redis.

        Args:
            symbol: Trading symbol
            data: SMC data to persist
        """
        if not self.redis_client or not self.redis_client.is_connected:
            return

        try:
            # Store order blocks (24h TTL)
            ob_key = f"smc:order_blocks:{symbol}"
            await self.redis_client.client.set(
                ob_key,
                json.dumps(data.get("order_blocks", [])).encode('utf-8'),
                ex=86400
            )

            # Store FVGs (24h TTL)
            fvg_key = f"smc:fvg:{symbol}"
            await self.redis_client.client.set(
                fvg_key,
                json.dumps(data.get("fair_value_gaps", [])).encode('utf-8'),
                ex=86400
            )

            # Store market structure (1h TTL - changes more frequently)
            structure_key = f"smc:structure:{symbol}"
            await self.redis_client.client.set(
                structure_key,
                json.dumps(data.get("market_structure", {})).encode('utf-8'),
                ex=3600
            )

            # Store liquidity pools (24h TTL)
            pools_key = f"smc:liquidity_pools:{symbol}"
            await self.redis_client.client.set(
                pools_key,
                json.dumps(data.get("liquidity_pools", [])).encode('utf-8'),
                ex=86400
            )

            # Store premium/discount (4h TTL)
            pd_key = f"smc:premium_discount:{symbol}"
            await self.redis_client.client.set(
                pd_key,
                json.dumps(data.get("premium_discount", {})).encode('utf-8'),
                ex=14400
            )

            logger.debug(f"Persisted SMC data for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to persist SMC data for {symbol}: {e}")

    # =========================================================================
    # Helper Methods for Parsing Cached Data
    # =========================================================================

    def _parse_order_blocks(self, ob_dicts: List[Dict]) -> List[OrderBlock]:
        """Parse order block dictionaries back to objects."""
        blocks = []
        for d in ob_dicts:
            try:
                timestamp = d.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                blocks.append(OrderBlock(
                    direction=SMCDirection(d.get("direction", "bullish")),
                    price_high=float(d.get("price_high", 0)),
                    price_low=float(d.get("price_low", 0)),
                    midpoint=float(d.get("midpoint", 0)),
                    impulse_size_pct=float(d.get("impulse_size_pct", 0)),
                    timeframe=d.get("timeframe", "4h"),
                    timestamp=timestamp,
                    tested=d.get("tested", False),
                    mitigated=d.get("mitigated", False),
                    test_count=d.get("test_count", 0),
                    strength=SMCStrength(d.get("strength", "moderate")),
                ))
            except Exception:
                continue
        return blocks

    def _parse_fvgs(self, fvg_dicts: List[Dict]) -> List[FairValueGap]:
        """Parse FVG dictionaries back to objects."""
        fvgs = []
        for d in fvg_dicts:
            try:
                timestamp = d.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                fvgs.append(FairValueGap(
                    direction=SMCDirection(d.get("direction", "bullish")),
                    gap_high=float(d.get("gap_high", 0)),
                    gap_low=float(d.get("gap_low", 0)),
                    midpoint=float(d.get("midpoint", 0)),
                    gap_size_pct=float(d.get("gap_size_pct", 0)),
                    timeframe=d.get("timeframe", "4h"),
                    timestamp=timestamp,
                    fill_pct=float(d.get("fill_pct", 0)),
                    filled=d.get("filled", False),
                    strength=SMCStrength(d.get("strength", "moderate")),
                ))
            except Exception:
                continue
        return fvgs

    def _parse_liquidity_pools(self, pool_dicts: List[Dict]) -> List[LiquidityPool]:
        """Parse liquidity pool dictionaries back to objects."""
        pools = []
        for d in pool_dicts:
            try:
                timestamp = d.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.utcnow()

                pools.append(LiquidityPool(
                    side=d.get("side", "buy"),
                    price_level=float(d.get("price_level", 0)),
                    source=d.get("source", "swing_high"),
                    num_touches=d.get("num_touches", 1),
                    timeframe=d.get("timeframe", "4h"),
                    timestamp=timestamp,
                    swept=d.get("swept", False),
                    strength=SMCStrength(d.get("strength", "moderate")),
                ))
            except Exception:
                continue
        return pools
