"""Chart Pattern Detection Service.

Detects classical chart patterns using scipy signal processing:
- Head and Shoulders (bearish reversal)
- Inverse Head and Shoulders (bullish reversal)
- Double Top / Double Bottom
- Ascending/Descending Triangles
- Bull/Bear Flags
- Rising/Falling Wedges

Reference: Technical Analysis patterns and scipy.signal.find_peaks
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    from scipy.signal import find_peaks
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. Pattern detection limited.")


class PatternType(Enum):
    """Types of chart patterns."""
    HEAD_SHOULDERS = "head_and_shoulders"
    INV_HEAD_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRIC_TRIANGLE = "symmetric_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"


class PatternSignal(Enum):
    """Signal direction from pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Pattern:
    """Detected chart pattern."""
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float
    start_index: int
    end_index: int
    target_price: Optional[float]
    stop_price: Optional[float]
    neckline: Optional[float]
    description: str
    key_levels: Dict[str, float]


class PatternDetector:
    """Detects classical chart patterns in OHLCV data.

    Uses scipy.signal.find_peaks for swing high/low detection
    and linear regression for trendline analysis.
    """

    def __init__(
        self,
        min_pattern_bars: int = 20,
        peak_distance: int = 5,
        peak_prominence: float = 0.01,
    ):
        """Initialize Pattern Detector.

        Args:
            min_pattern_bars: Minimum bars required for pattern detection
            peak_distance: Minimum distance between peaks (scipy find_peaks)
            peak_prominence: Minimum prominence for peak detection (as % of price)
        """
        self.min_pattern_bars = min_pattern_bars
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence

        if not HAS_SCIPY:
            logger.warning("scipy not available - using basic peak detection")

    def detect_all(self, df: pd.DataFrame) -> List[Pattern]:
        """Detect all patterns in the data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            List of detected patterns
        """
        if df.empty or len(df) < self.min_pattern_bars:
            return []

        patterns = []

        # Find swing highs and lows
        highs, lows = self._find_swings(df)

        # Detect each pattern type
        patterns.extend(self._detect_head_shoulders(df, highs, lows))
        patterns.extend(self._detect_double_patterns(df, highs, lows))
        patterns.extend(self._detect_triangles(df, highs, lows))
        patterns.extend(self._detect_flags(df, highs, lows))
        patterns.extend(self._detect_wedges(df, highs, lows))

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def _find_swings(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find swing highs and swing lows.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        high = df["high"].values
        low = df["low"].values

        if HAS_SCIPY:
            # Use scipy for robust peak detection
            prominence = self.peak_prominence * np.mean(high)

            swing_highs, _ = find_peaks(
                high,
                distance=self.peak_distance,
                prominence=prominence,
            )

            swing_lows, _ = find_peaks(
                -low,  # Invert to find valleys
                distance=self.peak_distance,
                prominence=prominence,
            )
        else:
            # Basic peak detection fallback
            swing_highs = self._basic_find_peaks(high)
            swing_lows = self._basic_find_peaks(-low)

        return swing_highs, swing_lows

    def _basic_find_peaks(self, data: np.ndarray) -> np.ndarray:
        """Basic peak detection without scipy."""
        peaks = []
        for i in range(self.peak_distance, len(data) - self.peak_distance):
            if data[i] == max(data[i-self.peak_distance:i+self.peak_distance+1]):
                peaks.append(i)
        return np.array(peaks)

    def _detect_head_shoulders(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[Pattern]:
        """Detect Head and Shoulders / Inverse Head and Shoulders patterns.

        H&S: high, higher-high (head), lower-high (right shoulder)
        Inverse: low, lower-low (head), higher-low (right shoulder)
        """
        patterns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Head and Shoulders (bearish)
        if len(highs) >= 3:
            for i in range(len(highs) - 2):
                left_shoulder = highs[i]
                head = highs[i + 1]
                right_shoulder = highs[i + 2]

                left_h = high[left_shoulder]
                head_h = high[head]
                right_h = high[right_shoulder]

                # Check pattern: head higher than both shoulders
                if head_h > left_h and head_h > right_h:
                    # Shoulders should be roughly equal (within 3%)
                    shoulder_diff = abs(left_h - right_h) / left_h
                    if shoulder_diff < 0.03:
                        # Find neckline (lowest low between shoulder peaks)
                        neckline_idx = slice(left_shoulder, right_shoulder + 1)
                        neckline = low[neckline_idx].min()

                        # Calculate target (measured move)
                        pattern_height = head_h - neckline
                        target = neckline - pattern_height

                        # Current price below neckline = pattern activated
                        current_price = close[-1]
                        if current_price < neckline:
                            confidence = 0.85
                            desc = "H&S ACTIVE: Price broke neckline"
                        else:
                            confidence = 0.70
                            desc = "H&S FORMING: Watching for neckline break"

                        patterns.append(Pattern(
                            pattern_type=PatternType.HEAD_SHOULDERS,
                            signal=PatternSignal.BEARISH,
                            confidence=confidence,
                            start_index=left_shoulder,
                            end_index=right_shoulder,
                            target_price=target,
                            stop_price=head_h * 1.01,  # Stop above head
                            neckline=neckline,
                            description=f"{desc}. Neckline: ${neckline:,.2f}, Target: ${target:,.2f}",
                            key_levels={
                                "left_shoulder": left_h,
                                "head": head_h,
                                "right_shoulder": right_h,
                                "neckline": neckline,
                            },
                        ))

        # Inverse Head and Shoulders (bullish)
        if len(lows) >= 3:
            for i in range(len(lows) - 2):
                left_shoulder = lows[i]
                head = lows[i + 1]
                right_shoulder = lows[i + 2]

                left_l = low[left_shoulder]
                head_l = low[head]
                right_l = low[right_shoulder]

                # Check pattern: head lower than both shoulders
                if head_l < left_l and head_l < right_l:
                    shoulder_diff = abs(left_l - right_l) / left_l
                    if shoulder_diff < 0.03:
                        # Find neckline (highest high between shoulder troughs)
                        neckline_idx = slice(left_shoulder, right_shoulder + 1)
                        neckline = high[neckline_idx].max()

                        pattern_height = neckline - head_l
                        target = neckline + pattern_height

                        current_price = close[-1]
                        if current_price > neckline:
                            confidence = 0.85
                            desc = "INV H&S ACTIVE: Price broke neckline"
                        else:
                            confidence = 0.70
                            desc = "INV H&S FORMING: Watching for neckline break"

                        patterns.append(Pattern(
                            pattern_type=PatternType.INV_HEAD_SHOULDERS,
                            signal=PatternSignal.BULLISH,
                            confidence=confidence,
                            start_index=left_shoulder,
                            end_index=right_shoulder,
                            target_price=target,
                            stop_price=head_l * 0.99,
                            neckline=neckline,
                            description=f"{desc}. Neckline: ${neckline:,.2f}, Target: ${target:,.2f}",
                            key_levels={
                                "left_shoulder": left_l,
                                "head": head_l,
                                "right_shoulder": right_l,
                                "neckline": neckline,
                            },
                        ))

        return patterns

    def _detect_double_patterns(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[Pattern]:
        """Detect Double Top and Double Bottom patterns."""
        patterns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Double Top (bearish)
        if len(highs) >= 2:
            for i in range(len(highs) - 1):
                first = highs[i]
                second = highs[i + 1]

                first_h = high[first]
                second_h = high[second]

                # Tops should be within 1.5% of each other
                if abs(first_h - second_h) / first_h < 0.015:
                    # Find support (lowest between peaks)
                    support = low[first:second + 1].min()

                    pattern_height = first_h - support
                    target = support - pattern_height

                    current_price = close[-1]
                    if current_price < support:
                        confidence = 0.80
                    else:
                        confidence = 0.60

                    patterns.append(Pattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        signal=PatternSignal.BEARISH,
                        confidence=confidence,
                        start_index=first,
                        end_index=second,
                        target_price=target,
                        stop_price=max(first_h, second_h) * 1.01,
                        neckline=support,
                        description=f"Double Top at ${first_h:,.2f}. Support: ${support:,.2f}",
                        key_levels={
                            "first_top": first_h,
                            "second_top": second_h,
                            "support": support,
                        },
                    ))

        # Double Bottom (bullish)
        if len(lows) >= 2:
            for i in range(len(lows) - 1):
                first = lows[i]
                second = lows[i + 1]

                first_l = low[first]
                second_l = low[second]

                if abs(first_l - second_l) / first_l < 0.015:
                    resistance = high[first:second + 1].max()

                    pattern_height = resistance - first_l
                    target = resistance + pattern_height

                    current_price = close[-1]
                    if current_price > resistance:
                        confidence = 0.80
                    else:
                        confidence = 0.60

                    patterns.append(Pattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        signal=PatternSignal.BULLISH,
                        confidence=confidence,
                        start_index=first,
                        end_index=second,
                        target_price=target,
                        stop_price=min(first_l, second_l) * 0.99,
                        neckline=resistance,
                        description=f"Double Bottom at ${first_l:,.2f}. Resistance: ${resistance:,.2f}",
                        key_levels={
                            "first_bottom": first_l,
                            "second_bottom": second_l,
                            "resistance": resistance,
                        },
                    ))

        return patterns

    def _detect_triangles(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[Pattern]:
        """Detect triangle patterns (ascending, descending, symmetric)."""
        patterns = []

        if len(highs) < 3 or len(lows) < 3:
            return patterns

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Get recent swing points
        recent_highs = highs[-4:] if len(highs) >= 4 else highs
        recent_lows = lows[-4:] if len(lows) >= 4 else lows

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return patterns

        # Calculate trendlines using linear regression
        high_prices = [high[i] for i in recent_highs]
        low_prices = [low[i] for i in recent_lows]

        if HAS_SCIPY and len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # High trendline slope
            high_slope, high_intercept, _, _, _ = linregress(
                range(len(high_prices)), high_prices
            )

            # Low trendline slope
            low_slope, low_intercept, _, _, _ = linregress(
                range(len(low_prices)), low_prices
            )

            # Classify triangle type
            high_flat = abs(high_slope) < 0.001 * np.mean(high_prices)
            low_flat = abs(low_slope) < 0.001 * np.mean(low_prices)
            converging = high_slope < 0 and low_slope > 0

            current_price = close[-1]

            if low_slope > 0 and high_flat:
                # Ascending triangle (bullish)
                resistance = np.mean(high_prices)
                target = resistance + (resistance - min(low_prices))

                patterns.append(Pattern(
                    pattern_type=PatternType.ASCENDING_TRIANGLE,
                    signal=PatternSignal.BULLISH,
                    confidence=0.75,
                    start_index=min(recent_highs[0], recent_lows[0]),
                    end_index=max(recent_highs[-1], recent_lows[-1]),
                    target_price=target,
                    stop_price=min(low_prices) * 0.99,
                    neckline=resistance,
                    description=f"Ascending Triangle. Resistance: ${resistance:,.2f}",
                    key_levels={
                        "resistance": resistance,
                        "rising_support": min(low_prices),
                    },
                ))

            elif high_slope < 0 and low_flat:
                # Descending triangle (bearish)
                support = np.mean(low_prices)
                target = support - (max(high_prices) - support)

                patterns.append(Pattern(
                    pattern_type=PatternType.DESCENDING_TRIANGLE,
                    signal=PatternSignal.BEARISH,
                    confidence=0.75,
                    start_index=min(recent_highs[0], recent_lows[0]),
                    end_index=max(recent_highs[-1], recent_lows[-1]),
                    target_price=target,
                    stop_price=max(high_prices) * 1.01,
                    neckline=support,
                    description=f"Descending Triangle. Support: ${support:,.2f}",
                    key_levels={
                        "support": support,
                        "falling_resistance": max(high_prices),
                    },
                ))

            elif converging:
                # Symmetric triangle (neutral until breakout)
                apex = (np.mean(high_prices) + np.mean(low_prices)) / 2

                patterns.append(Pattern(
                    pattern_type=PatternType.SYMMETRIC_TRIANGLE,
                    signal=PatternSignal.NEUTRAL,
                    confidence=0.65,
                    start_index=min(recent_highs[0], recent_lows[0]),
                    end_index=max(recent_highs[-1], recent_lows[-1]),
                    target_price=None,
                    stop_price=None,
                    neckline=apex,
                    description=f"Symmetric Triangle. Apex: ${apex:,.2f}",
                    key_levels={
                        "upper_trendline": max(high_prices),
                        "lower_trendline": min(low_prices),
                        "apex": apex,
                    },
                ))

        return patterns

    def _detect_flags(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[Pattern]:
        """Detect bull and bear flag patterns.

        Flags: Sharp move (pole) followed by consolidation (flag)
        """
        patterns = []

        if len(df) < 30:
            return patterns

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # Look for recent sharp move followed by consolidation
        lookback = 20
        recent_close = close[-lookback:]
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]

        # Calculate price change in first half vs second half
        first_half = recent_close[:lookback//2]
        second_half = recent_close[lookback//2:]

        pole_change = (first_half[-1] - first_half[0]) / first_half[0]
        flag_change = (second_half[-1] - second_half[0]) / second_half[0]

        # Flag volatility (should be lower than pole)
        pole_vol = np.std(first_half) / np.mean(first_half)
        flag_vol = np.std(second_half) / np.mean(second_half)

        # Bull Flag: Strong up move + tight consolidation
        if pole_change > 0.05 and abs(flag_change) < 0.02 and flag_vol < pole_vol:
            pole_height = recent_high[:lookback//2].max() - recent_low[:lookback//2].min()
            target = close[-1] + pole_height

            patterns.append(Pattern(
                pattern_type=PatternType.BULL_FLAG,
                signal=PatternSignal.BULLISH,
                confidence=0.70,
                start_index=len(df) - lookback,
                end_index=len(df) - 1,
                target_price=target,
                stop_price=recent_low[lookback//2:].min() * 0.99,
                neckline=None,
                description=f"Bull Flag. Pole: +{pole_change*100:.1f}%, Target: ${target:,.2f}",
                key_levels={
                    "pole_low": recent_low[:lookback//2].min(),
                    "pole_high": recent_high[:lookback//2].max(),
                    "flag_support": recent_low[lookback//2:].min(),
                },
            ))

        # Bear Flag: Strong down move + tight consolidation
        elif pole_change < -0.05 and abs(flag_change) < 0.02 and flag_vol < pole_vol:
            pole_height = recent_high[:lookback//2].max() - recent_low[:lookback//2].min()
            target = close[-1] - pole_height

            patterns.append(Pattern(
                pattern_type=PatternType.BEAR_FLAG,
                signal=PatternSignal.BEARISH,
                confidence=0.70,
                start_index=len(df) - lookback,
                end_index=len(df) - 1,
                target_price=target,
                stop_price=recent_high[lookback//2:].max() * 1.01,
                neckline=None,
                description=f"Bear Flag. Pole: {pole_change*100:.1f}%, Target: ${target:,.2f}",
                key_levels={
                    "pole_high": recent_high[:lookback//2].max(),
                    "pole_low": recent_low[:lookback//2].min(),
                    "flag_resistance": recent_high[lookback//2:].max(),
                },
            ))

        return patterns

    def _detect_wedges(
        self,
        df: pd.DataFrame,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[Pattern]:
        """Detect rising and falling wedge patterns.

        Rising Wedge: Higher highs and higher lows, but converging (bearish)
        Falling Wedge: Lower highs and lower lows, but converging (bullish)
        """
        patterns = []

        if not HAS_SCIPY or len(highs) < 3 or len(lows) < 3:
            return patterns

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        recent_highs = highs[-4:] if len(highs) >= 4 else highs
        recent_lows = lows[-4:] if len(lows) >= 4 else lows

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return patterns

        high_prices = [high[i] for i in recent_highs]
        low_prices = [low[i] for i in recent_lows]

        high_slope, _, _, _, _ = linregress(range(len(high_prices)), high_prices)
        low_slope, _, _, _, _ = linregress(range(len(low_prices)), low_prices)

        # Both trending up but converging = Rising Wedge (bearish)
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
            target = min(low_prices) - (max(high_prices) - min(low_prices)) * 0.5

            patterns.append(Pattern(
                pattern_type=PatternType.RISING_WEDGE,
                signal=PatternSignal.BEARISH,
                confidence=0.65,
                start_index=min(recent_highs[0], recent_lows[0]),
                end_index=max(recent_highs[-1], recent_lows[-1]),
                target_price=target,
                stop_price=max(high_prices) * 1.01,
                neckline=None,
                description=f"Rising Wedge (bearish). Target: ${target:,.2f}",
                key_levels={
                    "upper_trendline": max(high_prices),
                    "lower_trendline": min(low_prices),
                },
            ))

        # Both trending down but converging = Falling Wedge (bullish)
        elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            target = max(high_prices) + (max(high_prices) - min(low_prices)) * 0.5

            patterns.append(Pattern(
                pattern_type=PatternType.FALLING_WEDGE,
                signal=PatternSignal.BULLISH,
                confidence=0.65,
                start_index=min(recent_highs[0], recent_lows[0]),
                end_index=max(recent_highs[-1], recent_lows[-1]),
                target_price=target,
                stop_price=min(low_prices) * 0.99,
                neckline=None,
                description=f"Falling Wedge (bullish). Target: ${target:,.2f}",
                key_levels={
                    "upper_trendline": max(high_prices),
                    "lower_trendline": min(low_prices),
                },
            ))

        return patterns

    def get_pattern_summary(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """Get summary of detected patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            Summary dictionary
        """
        if not patterns:
            return {
                "count": 0,
                "bullish": 0,
                "bearish": 0,
                "neutral": 0,
                "top_pattern": None,
            }

        bullish = [p for p in patterns if p.signal == PatternSignal.BULLISH]
        bearish = [p for p in patterns if p.signal == PatternSignal.BEARISH]
        neutral = [p for p in patterns if p.signal == PatternSignal.NEUTRAL]

        top = patterns[0] if patterns else None

        return {
            "count": len(patterns),
            "bullish": len(bullish),
            "bearish": len(bearish),
            "neutral": len(neutral),
            "top_pattern": {
                "type": top.pattern_type.value,
                "signal": top.signal.value,
                "confidence": top.confidence,
                "description": top.description,
            } if top else None,
            "net_bias": (len(bullish) - len(bearish)) / max(len(patterns), 1),
        }
