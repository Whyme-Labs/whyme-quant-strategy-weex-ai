"""Trend Following Strategy Agent.

Based on research insights:
- Turtle Traders ($175M): "Buy when price closes above 20-day high"
- VCP Pattern (@felipeguirao): Volatility contraction before breakout
- EMA alignment (8, 20, 50) for trend confirmation
- "Wins less often, but winners are much larger than losers"
- Key: Take EVERY trade, avoid taking profits too early
- Strict stop-losses protect against reversals
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_STRATEGY_GENERATION


class BreakoutType(Enum):
    """Type of breakout signal."""
    CHANNEL_BREAKOUT = "channel_breakout"  # Turtle-style
    VCP_BREAKOUT = "vcp_breakout"  # Volatility contraction pattern
    EMA_CROSSOVER = "ema_crossover"  # Moving average crossover
    MOMENTUM_SURGE = "momentum_surge"  # Strong momentum


@dataclass
class TrendSignal:
    """Trend following trading signal."""
    direction: str  # "long" or "short"
    breakout_type: BreakoutType
    entry_price: float
    stop_price: float  # Strict stops are crucial
    initial_target: Optional[float]  # May trail instead
    position_size_pct: float
    reasoning: str


class TrendFollowingAgent(BaseAgent):
    """Trend Following Strategy Agent.

    Strategy Philosophy:
    - Buy high, sell higher (or short low, cover lower)
    - Trends persist longer than expected
    - Low win rate, high reward per winner
    - MUST take every signal (small number account for bulk of profits)
    - NEVER take profits too early - let winners run

    Entry Signals:
    - 20-day channel breakout (Turtle system)
    - VCP pattern breakout (7+ candle consolidation, higher lows)
    - EMA alignment (8 > 20 > 50) with new high
    - Momentum surge with volume

    Exit Rules:
    - Strict stop-loss below recent swing low (longs)
    - Trailing stop using ATR or recent low
    - Exit only when trend clearly reverses
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize trend following agent.

        Args:
            config: Configuration with:
                - channel_period: Donchian channel period (default 20)
                - ema_periods: EMA periods (default [8, 20, 50])
                - atr_period: ATR period for stops (default 14)
                - atr_multiplier: Stop distance in ATRs (default 2)
                - max_position_pct: Maximum position (default 0.1)
                - vcp_min_candles: Minimum candles for VCP (default 7)
        """
        super().__init__(config)
        self.channel_period = config.get("channel_period", 20)
        self.ema_periods = config.get("ema_periods", [8, 20, 50])
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.max_position_pct = config.get("max_position_pct", 0.1)
        self.vcp_min_candles = config.get("vcp_min_candles", 7)

        self.price_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.volume_history: List[float] = []

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate trend following signals.

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
        volume = market_data.get("volume", 0)

        # Update history
        if current_price > 0:
            self.price_history.append(current_price)
            self.high_history.append(high)
            self.low_history.append(low)
            self.volume_history.append(volume)

            # Keep limited history
            max_len = max(self.channel_period, max(self.ema_periods)) * 2
            if len(self.price_history) > max_len:
                self.price_history = self.price_history[-max_len:]
                self.high_history = self.high_history[-max_len:]
                self.low_history = self.low_history[-max_len:]
                self.volume_history = self.volume_history[-max_len:]

        # Check if we have enough data
        min_required = max(self.channel_period, max(self.ema_periods))
        if len(self.price_history) < min_required:
            return {
                "signal": None,
                "reasoning": f"Insufficient data: {len(self.price_history)}/{min_required} periods"
            }

        # Generate signals
        signal = self._generate_signal(current_price, regime)

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_STRATEGY_GENERATION,
            model="trend_following_v1",
            input_data={
                "price": current_price,
                "high": high,
                "low": low,
                "volume": volume,
                "regime": regime,
            },
            output_data={
                "signal_direction": signal.direction if signal else None,
                "breakout_type": signal.breakout_type.value if signal else None,
                "stop_price": signal.stop_price if signal else None,
                "position_size_pct": signal.position_size_pct if signal else 0,
            },
            explanation=signal.reasoning if signal else "No trend following signal",
        )

        if signal and signal.direction != "none":
            return {
                "signal": {
                    "strategy": "trend_following",
                    "direction": signal.direction,
                    "breakout_type": signal.breakout_type.value,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "initial_target": signal.initial_target,
                    "position_size_pct": signal.position_size_pct,
                },
                "reasoning": signal.reasoning,
            }
        else:
            return {
                "signal": None,
                "reasoning": signal.reasoning if signal else "No signal"
            }

    def _generate_signal(
        self,
        current_price: float,
        regime: Dict[str, Any],
    ) -> Optional[TrendSignal]:
        """Generate trend following signal.

        Checks multiple breakout types and returns the strongest signal.

        Args:
            current_price: Current price
            regime: Market regime

        Returns:
            TrendSignal or None
        """
        prices = np.array(self.price_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)

        # Calculate key levels
        channel_high = np.max(highs[-self.channel_period:])
        channel_low = np.min(lows[-self.channel_period:])

        # Calculate EMAs
        ema_8 = self._calculate_ema(prices, 8)
        ema_20 = self._calculate_ema(prices, 20)
        ema_50 = self._calculate_ema(prices, min(50, len(prices)))

        # Calculate ATR for stop placement
        atr = self._calculate_atr(highs, lows, prices)

        signals = []

        # 1. Turtle-style Channel Breakout
        channel_signal = self._check_channel_breakout(
            current_price, channel_high, channel_low, atr
        )
        if channel_signal:
            signals.append(channel_signal)

        # 2. EMA Crossover/Alignment
        ema_signal = self._check_ema_alignment(
            current_price, ema_8, ema_20, ema_50, atr, lows
        )
        if ema_signal:
            signals.append(ema_signal)

        # 3. VCP Pattern (simplified)
        vcp_signal = self._check_vcp_pattern(
            prices, highs, lows, current_price, atr
        )
        if vcp_signal:
            signals.append(vcp_signal)

        # Return strongest signal (by position size)
        if signals:
            return max(signals, key=lambda s: s.position_size_pct)

        return TrendSignal(
            direction="none",
            breakout_type=BreakoutType.CHANNEL_BREAKOUT,
            entry_price=current_price,
            stop_price=0,
            initial_target=None,
            position_size_pct=0,
            reasoning=f"No breakout. Price at {current_price:.2f}, Channel: {channel_low:.2f}-{channel_high:.2f}"
        )

    def _check_channel_breakout(
        self,
        current_price: float,
        channel_high: float,
        channel_low: float,
        atr: float,
    ) -> Optional[TrendSignal]:
        """Check for Turtle-style channel breakout.

        Args:
            current_price: Current price
            channel_high: N-day high
            channel_low: N-day low
            atr: Average True Range

        Returns:
            TrendSignal if breakout detected
        """
        # Long breakout: price above channel high
        if current_price > channel_high:
            stop_price = current_price - (self.atr_multiplier * atr)
            return TrendSignal(
                direction="long",
                breakout_type=BreakoutType.CHANNEL_BREAKOUT,
                entry_price=current_price,
                stop_price=stop_price,
                initial_target=None,  # Let it run
                position_size_pct=self.max_position_pct,
                reasoning=f"Channel breakout LONG: Price {current_price:.2f} > {self.channel_period}-day high {channel_high:.2f}. Stop at {stop_price:.2f} ({self.atr_multiplier}x ATR)"
            )

        # Short breakout: price below channel low
        if current_price < channel_low:
            stop_price = current_price + (self.atr_multiplier * atr)
            return TrendSignal(
                direction="short",
                breakout_type=BreakoutType.CHANNEL_BREAKOUT,
                entry_price=current_price,
                stop_price=stop_price,
                initial_target=None,
                position_size_pct=self.max_position_pct * 0.8,  # Slightly less for shorts
                reasoning=f"Channel breakdown SHORT: Price {current_price:.2f} < {self.channel_period}-day low {channel_low:.2f}. Stop at {stop_price:.2f}"
            )

        return None

    def _check_ema_alignment(
        self,
        current_price: float,
        ema_8: float,
        ema_20: float,
        ema_50: float,
        atr: float,
        lows: np.ndarray,
    ) -> Optional[TrendSignal]:
        """Check for EMA alignment signal.

        Bullish: price > EMA8 > EMA20 > EMA50
        Bearish: price < EMA8 < EMA20 < EMA50

        Args:
            current_price: Current price
            ema_8: 8-period EMA
            ema_20: 20-period EMA
            ema_50: 50-period EMA
            atr: Average True Range
            lows: Low prices for stop placement

        Returns:
            TrendSignal if alignment detected
        """
        # Bullish alignment
        if current_price > ema_8 > ema_20 > ema_50:
            # Use recent swing low as stop
            recent_low = np.min(lows[-5:])
            stop_price = min(recent_low, current_price - (self.atr_multiplier * atr))

            return TrendSignal(
                direction="long",
                breakout_type=BreakoutType.EMA_CROSSOVER,
                entry_price=current_price,
                stop_price=stop_price,
                initial_target=None,
                position_size_pct=self.max_position_pct * 0.7,
                reasoning=f"Bullish EMA alignment: {current_price:.2f} > EMA8 ({ema_8:.2f}) > EMA20 ({ema_20:.2f}) > EMA50 ({ema_50:.2f}). Stop at {stop_price:.2f}"
            )

        # Bearish alignment
        if current_price < ema_8 < ema_20 < ema_50:
            recent_high = np.max(lows[-5:])  # Using lows array as proxy
            stop_price = max(recent_high, current_price + (self.atr_multiplier * atr))

            return TrendSignal(
                direction="short",
                breakout_type=BreakoutType.EMA_CROSSOVER,
                entry_price=current_price,
                stop_price=stop_price,
                initial_target=None,
                position_size_pct=self.max_position_pct * 0.5,
                reasoning=f"Bearish EMA alignment: {current_price:.2f} < EMA8 < EMA20 < EMA50. Stop at {stop_price:.2f}"
            )

        return None

    def _check_vcp_pattern(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        current_price: float,
        atr: float,
    ) -> Optional[TrendSignal]:
        """Check for Volatility Contraction Pattern (VCP).

        VCP criteria:
        - Minimum 7 candles of consolidation
        - Decreasing volatility (tightening range)
        - Higher lows forming
        - Breakout above consolidation high

        Args:
            prices: Price history
            highs: High prices
            lows: Low prices
            current_price: Current price
            atr: Average True Range

        Returns:
            TrendSignal if VCP breakout detected
        """
        if len(prices) < self.vcp_min_candles + 5:
            return None

        # Get recent consolidation period
        consolidation_highs = highs[-self.vcp_min_candles:-1]
        consolidation_lows = lows[-self.vcp_min_candles:-1]

        # Check for tightening range (volatility contraction)
        first_half_range = np.max(consolidation_highs[:len(consolidation_highs)//2]) - np.min(consolidation_lows[:len(consolidation_lows)//2])
        second_half_range = np.max(consolidation_highs[len(consolidation_highs)//2:]) - np.min(consolidation_lows[len(consolidation_lows)//2:])

        volatility_contracting = second_half_range < first_half_range * 0.8

        # Check for higher lows (bullish structure)
        recent_lows = consolidation_lows[-3:]
        higher_lows = all(recent_lows[i] >= recent_lows[i-1] * 0.99 for i in range(1, len(recent_lows)))

        # Consolidation high (resistance level)
        consolidation_high = np.max(consolidation_highs)

        # Breakout condition
        breakout = current_price > consolidation_high

        if volatility_contracting and higher_lows and breakout:
            stop_price = np.min(consolidation_lows[-3:])

            return TrendSignal(
                direction="long",
                breakout_type=BreakoutType.VCP_BREAKOUT,
                entry_price=current_price,
                stop_price=stop_price,
                initial_target=None,  # Let it run
                position_size_pct=self.max_position_pct * 0.9,  # High confidence pattern
                reasoning=f"VCP Breakout: {self.vcp_min_candles} candle consolidation with tightening range and higher lows. Breakout above {consolidation_high:.2f}. Stop at {stop_price:.2f}"
            )

        return None

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> float:
        """Calculate Average True Range."""
        if len(highs) < 2:
            return highs[0] - lows[0] if len(highs) > 0 else 0

        period = min(self.atr_period, len(highs) - 1)

        true_ranges = []
        for i in range(1, period + 1):
            idx = -i
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1])
            )
            true_ranges.append(tr)

        return np.mean(true_ranges) if true_ranges else 0
