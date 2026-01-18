"""Pivot Point Strategy Agent.

Based on classic floor trader techniques:
- Pivot points are key support/resistance levels
- Calculated from previous period's High, Low, Close
- S1, S2, S3 (support) and R1, R2, R3 (resistance)
- Price often bounces or breaks these levels
- Works well in ranging and trending markets
- Provides clear entry/exit levels
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_STRATEGY_GENERATION


class PivotSignalType(Enum):
    """Type of pivot signal."""
    SUPPORT_BOUNCE = "support_bounce"  # Price bouncing off support
    RESISTANCE_REJECT = "resistance_reject"  # Price rejected at resistance
    BREAKOUT_ABOVE = "breakout_above"  # Breakout above resistance
    BREAKDOWN_BELOW = "breakdown_below"  # Breakdown below support
    PIVOT_BOUNCE = "pivot_bounce"  # Bounce off main pivot


@dataclass
class PivotLevels:
    """Calculated pivot levels."""
    pivot: float  # Main pivot point
    r1: float  # Resistance 1
    r2: float  # Resistance 2
    r3: float  # Resistance 3
    s1: float  # Support 1
    s2: float  # Support 2
    s3: float  # Support 3


@dataclass
class PivotSignal:
    """Pivot trading signal."""
    direction: str  # "long", "short", or "none"
    signal_type: PivotSignalType
    entry_price: float
    stop_price: float
    target_price: float
    position_size_pct: float
    pivot_level: str  # Which pivot level triggered
    reasoning: str


class PivotAgent(BaseAgent):
    """Pivot Point Strategy Agent.

    Strategy Philosophy:
    - Pivot points are self-fulfilling prophecy (many traders watch them)
    - Support becomes resistance and vice versa
    - Clear risk:reward at pivot levels
    - Combine with price action for confirmation

    Entry Signals:
    - Bounce: Price touches S1/S2/R1/R2 and reverses
    - Breakout: Price closes above R1/R2 with momentum
    - Breakdown: Price closes below S1/S2 with momentum
    - Pivot test: Price tests main pivot and bounces

    Exit Rules:
    - Target at next pivot level
    - Stop below/above entry pivot level
    - Trail stop as price moves in favor
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize pivot agent.

        Args:
            config: Configuration with:
                - bounce_threshold: Price distance for bounce detection (default 0.002)
                - breakout_threshold: Breakout confirmation threshold (default 0.005)
                - max_position_pct: Maximum position (default 0.06)
                - use_fibonacci: Use Fibonacci pivots instead of standard (default False)
        """
        super().__init__(config)
        self.bounce_threshold = config.get("bounce_threshold", 0.002)  # 0.2%
        self.breakout_threshold = config.get("breakout_threshold", 0.005)  # 0.5%
        self.max_position_pct = config.get("max_position_pct", 0.06)
        self.use_fibonacci = config.get("use_fibonacci", False)
        self.atr_multiplier = config.get("atr_multiplier", 1.5)

        # Daily OHLC for pivot calculation
        self.daily_high: Optional[float] = None
        self.daily_low: Optional[float] = None
        self.daily_close: Optional[float] = None

        self.price_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []

        # Cached pivot levels
        self._pivot_levels: Optional[PivotLevels] = None

    def set_daily_ohlc(self, high: float, low: float, close: float):
        """Set previous day's OHLC for pivot calculation.

        Args:
            high: Previous day high
            low: Previous day low
            close: Previous day close
        """
        self.daily_high = high
        self.daily_low = low
        self.daily_close = close
        self._pivot_levels = self._calculate_pivots(high, low, close)

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate pivot signals.

        Args:
            context: Dictionary containing:
                - market_data: Current market data
                - regime: Current market regime

        Returns:
            Dictionary with signal details
        """
        market_data = context.get("market_data", {})
        regime = context.get("regime", {})

        current_price = market_data.get("price", 0)
        high = market_data.get("high_24h", current_price)
        low = market_data.get("low_24h", current_price)

        # Update history
        if current_price > 0:
            self.price_history.append(current_price)
            self.high_history.append(high)
            self.low_history.append(low)

            # Keep limited history
            max_len = 100
            if len(self.price_history) > max_len:
                self.price_history = self.price_history[-max_len:]
                self.high_history = self.high_history[-max_len:]
                self.low_history = self.low_history[-max_len:]

        # Auto-calculate pivots from 24h data if not set
        if self._pivot_levels is None and len(self.price_history) > 24:
            self._auto_calculate_pivots()

        if self._pivot_levels is None:
            return {
                "signal": None,
                "reasoning": "Pivot levels not yet calculated. Need previous period data."
            }

        # Check if we have enough data
        if len(self.price_history) < 5:
            return {
                "signal": None,
                "reasoning": f"Insufficient data: {len(self.price_history)}/5 periods"
            }

        # Generate signals
        signal = self._generate_signal(current_price, regime)

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_STRATEGY_GENERATION,
            model="pivot_v1",
            input_data={
                "price": current_price,
                "high": high,
                "low": low,
                "pivot_levels": {
                    "pivot": self._pivot_levels.pivot,
                    "r1": self._pivot_levels.r1,
                    "r2": self._pivot_levels.r2,
                    "s1": self._pivot_levels.s1,
                    "s2": self._pivot_levels.s2,
                },
                "regime": regime,
            },
            output_data={
                "signal_direction": signal.direction if signal else None,
                "signal_type": signal.signal_type.value if signal else None,
                "pivot_level": signal.pivot_level if signal else None,
                "target_price": signal.target_price if signal else None,
                "position_size_pct": signal.position_size_pct if signal else 0,
            },
            explanation=signal.reasoning if signal else "No pivot signal",
        )

        if signal and signal.direction != "none":
            return {
                "signal": {
                    "strategy": "pivot",
                    "direction": signal.direction,
                    "signal_type": signal.signal_type.value,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "position_size_pct": signal.position_size_pct,
                    "pivot_level": signal.pivot_level,
                },
                "reasoning": signal.reasoning,
            }
        else:
            return {
                "signal": None,
                "reasoning": signal.reasoning if signal else "No signal"
            }

    def _auto_calculate_pivots(self):
        """Auto-calculate pivots from recent price history."""
        if len(self.high_history) < 24:
            return

        # Use last 24 hours as "previous period"
        period_high = max(self.high_history[-24:])
        period_low = min(self.low_history[-24:])
        period_close = self.price_history[-24]  # Close 24 hours ago

        self._pivot_levels = self._calculate_pivots(period_high, period_low, period_close)

    def _calculate_pivots(self, high: float, low: float, close: float) -> PivotLevels:
        """Calculate pivot point levels.

        Standard Pivot Formula:
        - Pivot = (High + Low + Close) / 3
        - R1 = 2 * Pivot - Low
        - R2 = Pivot + (High - Low)
        - R3 = High + 2 * (Pivot - Low)
        - S1 = 2 * Pivot - High
        - S2 = Pivot - (High - Low)
        - S3 = Low - 2 * (High - Pivot)

        Fibonacci Pivot Formula:
        - Pivot = (High + Low + Close) / 3
        - R1 = Pivot + 0.382 * (High - Low)
        - R2 = Pivot + 0.618 * (High - Low)
        - R3 = Pivot + 1.0 * (High - Low)
        - S1 = Pivot - 0.382 * (High - Low)
        - S2 = Pivot - 0.618 * (High - Low)
        - S3 = Pivot - 1.0 * (High - Low)

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            PivotLevels object
        """
        pivot = (high + low + close) / 3
        range_hl = high - low

        if self.use_fibonacci:
            r1 = pivot + 0.382 * range_hl
            r2 = pivot + 0.618 * range_hl
            r3 = pivot + 1.0 * range_hl
            s1 = pivot - 0.382 * range_hl
            s2 = pivot - 0.618 * range_hl
            s3 = pivot - 1.0 * range_hl
        else:
            r1 = 2 * pivot - low
            r2 = pivot + range_hl
            r3 = high + 2 * (pivot - low)
            s1 = 2 * pivot - high
            s2 = pivot - range_hl
            s3 = low - 2 * (high - pivot)

        return PivotLevels(
            pivot=pivot,
            r1=r1, r2=r2, r3=r3,
            s1=s1, s2=s2, s3=s3
        )

    def _generate_signal(
        self,
        current_price: float,
        regime: Dict[str, Any],
    ) -> Optional[PivotSignal]:
        """Generate pivot signal.

        Args:
            current_price: Current price
            regime: Market regime

        Returns:
            PivotSignal
        """
        pivots = self._pivot_levels
        prices = np.array(self.price_history)

        # Get price momentum direction
        if len(prices) >= 3:
            recent_direction = "up" if prices[-1] > prices[-3] else "down"
        else:
            recent_direction = "neutral"

        signals = []

        # 1. Check support bounces (long signals)
        support_signal = self._check_support_bounce(current_price, pivots, prices, recent_direction)
        if support_signal:
            signals.append(support_signal)

        # 2. Check resistance rejects (short signals)
        resistance_signal = self._check_resistance_reject(current_price, pivots, prices, recent_direction)
        if resistance_signal:
            signals.append(resistance_signal)

        # 3. Check breakouts above resistance
        breakout_signal = self._check_breakout_above(current_price, pivots, prices)
        if breakout_signal:
            signals.append(breakout_signal)

        # 4. Check breakdowns below support
        breakdown_signal = self._check_breakdown_below(current_price, pivots, prices)
        if breakdown_signal:
            signals.append(breakdown_signal)

        # 5. Check main pivot bounce
        pivot_signal = self._check_pivot_bounce(current_price, pivots, prices, recent_direction)
        if pivot_signal:
            signals.append(pivot_signal)

        # Return best signal (prioritize by position size which reflects confidence)
        if signals:
            return max(signals, key=lambda s: s.position_size_pct)

        return PivotSignal(
            direction="none",
            signal_type=PivotSignalType.SUPPORT_BOUNCE,
            entry_price=current_price,
            stop_price=0,
            target_price=0,
            position_size_pct=0,
            pivot_level="none",
            reasoning=f"No pivot signal. Price at {current_price:.2f}, Pivot: {pivots.pivot:.2f}, S1: {pivots.s1:.2f}, R1: {pivots.r1:.2f}"
        )

    def _check_support_bounce(
        self,
        current_price: float,
        pivots: PivotLevels,
        prices: np.ndarray,
        direction: str,
    ) -> Optional[PivotSignal]:
        """Check for bounce off support levels."""
        supports = [
            ("S1", pivots.s1, pivots.pivot),
            ("S2", pivots.s2, pivots.s1),
            ("S3", pivots.s3, pivots.s2),
        ]

        for level_name, support, target in supports:
            if support <= 0:
                continue

            distance_pct = (current_price - support) / support

            # Price near support and turning up
            if 0 < distance_pct < self.bounce_threshold and direction == "up":
                stop_price = support * (1 - self.bounce_threshold)

                return PivotSignal(
                    direction="long",
                    signal_type=PivotSignalType.SUPPORT_BOUNCE,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target,
                    position_size_pct=self.max_position_pct * 0.8,
                    pivot_level=level_name,
                    reasoning=f"Support Bounce LONG at {level_name}: Price {current_price:.2f} bouncing from {support:.2f}. Target: {target:.2f}"
                )

        return None

    def _check_resistance_reject(
        self,
        current_price: float,
        pivots: PivotLevels,
        prices: np.ndarray,
        direction: str,
    ) -> Optional[PivotSignal]:
        """Check for rejection at resistance levels."""
        resistances = [
            ("R1", pivots.r1, pivots.pivot),
            ("R2", pivots.r2, pivots.r1),
            ("R3", pivots.r3, pivots.r2),
        ]

        for level_name, resistance, target in resistances:
            if resistance <= 0:
                continue

            distance_pct = (resistance - current_price) / resistance

            # Price near resistance and turning down
            if 0 < distance_pct < self.bounce_threshold and direction == "down":
                stop_price = resistance * (1 + self.bounce_threshold)

                return PivotSignal(
                    direction="short",
                    signal_type=PivotSignalType.RESISTANCE_REJECT,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target,
                    position_size_pct=self.max_position_pct * 0.7,
                    pivot_level=level_name,
                    reasoning=f"Resistance Reject SHORT at {level_name}: Price {current_price:.2f} rejected from {resistance:.2f}. Target: {target:.2f}"
                )

        return None

    def _check_breakout_above(
        self,
        current_price: float,
        pivots: PivotLevels,
        prices: np.ndarray,
    ) -> Optional[PivotSignal]:
        """Check for breakout above resistance."""
        resistances = [
            ("R1", pivots.r1, pivots.r2),
            ("R2", pivots.r2, pivots.r3),
        ]

        for level_name, resistance, target in resistances:
            if resistance <= 0:
                continue

            # Breakout: Price closed above resistance with momentum
            breakout_pct = (current_price - resistance) / resistance

            if breakout_pct > self.breakout_threshold:
                # Confirm with prior prices below resistance
                if len(prices) >= 3 and prices[-3] < resistance:
                    stop_price = resistance * (1 - self.bounce_threshold)

                    return PivotSignal(
                        direction="long",
                        signal_type=PivotSignalType.BREAKOUT_ABOVE,
                        entry_price=current_price,
                        stop_price=stop_price,
                        target_price=target,
                        position_size_pct=self.max_position_pct,
                        pivot_level=level_name,
                        reasoning=f"Breakout LONG above {level_name}: Price {current_price:.2f} broke above {resistance:.2f}. Target: {target:.2f}"
                    )

        return None

    def _check_breakdown_below(
        self,
        current_price: float,
        pivots: PivotLevels,
        prices: np.ndarray,
    ) -> Optional[PivotSignal]:
        """Check for breakdown below support."""
        supports = [
            ("S1", pivots.s1, pivots.s2),
            ("S2", pivots.s2, pivots.s3),
        ]

        for level_name, support, target in supports:
            if support <= 0:
                continue

            # Breakdown: Price closed below support with momentum
            breakdown_pct = (support - current_price) / support

            if breakdown_pct > self.breakout_threshold:
                # Confirm with prior prices above support
                if len(prices) >= 3 and prices[-3] > support:
                    stop_price = support * (1 + self.bounce_threshold)

                    return PivotSignal(
                        direction="short",
                        signal_type=PivotSignalType.BREAKDOWN_BELOW,
                        entry_price=current_price,
                        stop_price=stop_price,
                        target_price=target,
                        position_size_pct=self.max_position_pct * 0.9,
                        pivot_level=level_name,
                        reasoning=f"Breakdown SHORT below {level_name}: Price {current_price:.2f} broke below {support:.2f}. Target: {target:.2f}"
                    )

        return None

    def _check_pivot_bounce(
        self,
        current_price: float,
        pivots: PivotLevels,
        prices: np.ndarray,
        direction: str,
    ) -> Optional[PivotSignal]:
        """Check for bounce off main pivot point."""
        pivot = pivots.pivot

        if pivot <= 0:
            return None

        distance_pct = abs(current_price - pivot) / pivot

        # Price near pivot
        if distance_pct < self.bounce_threshold:
            # Bullish bounce (price above pivot, turning up)
            if current_price > pivot and direction == "up":
                stop_price = pivots.s1
                target_price = pivots.r1

                return PivotSignal(
                    direction="long",
                    signal_type=PivotSignalType.PIVOT_BOUNCE,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    position_size_pct=self.max_position_pct * 0.6,
                    pivot_level="Pivot",
                    reasoning=f"Pivot Bounce LONG: Price {current_price:.2f} bouncing above pivot {pivot:.2f}. Target: R1 at {target_price:.2f}"
                )

            # Bearish bounce (price below pivot, turning down)
            if current_price < pivot and direction == "down":
                stop_price = pivots.r1
                target_price = pivots.s1

                return PivotSignal(
                    direction="short",
                    signal_type=PivotSignalType.PIVOT_BOUNCE,
                    entry_price=current_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    position_size_pct=self.max_position_pct * 0.5,
                    pivot_level="Pivot",
                    reasoning=f"Pivot Bounce SHORT: Price {current_price:.2f} rejected below pivot {pivot:.2f}. Target: S1 at {target_price:.2f}"
                )

        return None

    def get_pivot_levels(self) -> Optional[Dict[str, float]]:
        """Get current pivot levels.

        Returns:
            Dictionary of pivot levels or None
        """
        if self._pivot_levels is None:
            return None

        return {
            "pivot": self._pivot_levels.pivot,
            "r1": self._pivot_levels.r1,
            "r2": self._pivot_levels.r2,
            "r3": self._pivot_levels.r3,
            "s1": self._pivot_levels.s1,
            "s2": self._pivot_levels.s2,
            "s3": self._pivot_levels.s3,
        }
