"""Momentum Strategy Agent.

Based on research insights:
- "Momentum is the tendency for assets to continue moving in their recent direction"
- Rate of Change (ROC) measures price velocity
- RSI divergence confirms exhaustion
- Volume confirms conviction
- Works best in trending markets, not ranging
- Key: Trade WITH momentum, not against it
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_STRATEGY_GENERATION

if TYPE_CHECKING:
    from ..services.indicators_service import IndicatorsService
    from ..services.market_data_service import MarketDataService


class MomentumType(Enum):
    """Type of momentum signal."""
    ROC_BREAKOUT = "roc_breakout"  # Rate of change exceeds threshold
    RSI_MOMENTUM = "rsi_momentum"  # RSI showing strong momentum
    VOLUME_SURGE = "volume_surge"  # Price + volume acceleration
    MACD_CROSSOVER = "macd_crossover"  # MACD momentum shift


@dataclass
class MomentumSignal:
    """Momentum trading signal."""
    direction: str  # "long", "short", or "none"
    momentum_type: MomentumType
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    position_size_pct: float
    momentum_score: float  # 0-100 momentum strength
    reasoning: str


class MomentumAgent(BaseAgent):
    """Momentum Strategy Agent.

    Strategy Philosophy:
    - "An object in motion tends to stay in motion"
    - Price momentum persists in short to medium term
    - Strong momentum often precedes continuation
    - Volume confirms momentum conviction
    - Trade only in direction of momentum

    Entry Signals:
    - ROC > threshold with rising trend
    - RSI > 60 (bullish) or < 40 (bearish) with price confirmation
    - Volume surge (2x average) with price breakout
    - MACD crossover with histogram expansion

    Exit Rules:
    - Momentum exhaustion (RSI divergence)
    - ROC turning against position
    - Volume declining while price stalls
    """

    def __init__(
        self,
        config: Dict[str, Any],
        indicators_service: Optional["IndicatorsService"] = None,
        market_data_service: Optional["MarketDataService"] = None,
    ):
        """Initialize momentum agent.

        Args:
            config: Configuration with:
                - roc_period: Rate of change period (default 14)
                - roc_threshold: ROC breakout threshold (default 5%)
                - rsi_period: RSI period (default 14)
                - volume_mult: Volume surge multiplier (default 2.0)
                - macd_fast: MACD fast period (default 12)
                - macd_slow: MACD slow period (default 26)
                - macd_signal: MACD signal period (default 9)
                - max_position_pct: Maximum position (default 0.08)
                - timeframe: Candle timeframe (default "1h")
            indicators_service: Centralized indicator service
            market_data_service: Market data service for candles
        """
        super().__init__(config)
        self.roc_period = config.get("roc_period", 14)
        self.roc_threshold = config.get("roc_threshold", 0.05)  # 5%
        self.rsi_period = config.get("rsi_period", 14)
        self.volume_mult = config.get("volume_mult", 2.0)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        self.max_position_pct = config.get("max_position_pct", 0.08)
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.timeframe = config.get("timeframe", "1h")

        # Service dependencies
        self.indicators_service = indicators_service
        self.market_data_service = market_data_service

        self.price_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.volume_history: List[float] = []

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate momentum signals.

        Args:
            context: Dictionary containing:
                - market_data: Current market data with OHLCV candles
                - regime: Current market regime

        Returns:
            Dictionary with signal details
        """
        market_data = context.get("market_data", {})
        regime = context.get("regime", {})

        current_price = market_data.get("price", 0)
        volume = market_data.get("volume", 0)

        # Use OHLCV candle data for proper calculations (prefer 1h for momentum)
        candles = market_data.get("candles_1h") or market_data.get("candles_4h") or []

        # Initialize high/low for AI logging
        high = market_data.get("high_24h", current_price)
        low = market_data.get("low_24h", current_price)

        if candles and len(candles) >= 5:
            # Extract OHLCV arrays from candles
            self.price_history = [float(c.get("close", 0)) for c in candles]
            self.high_history = [float(c.get("high", 0)) for c in candles]
            self.low_history = [float(c.get("low", 0)) for c in candles]
            self.volume_history = [float(c.get("volume", 0)) for c in candles]
            # Update high/low from latest candle
            if candles:
                high = float(candles[-1].get("high", high))
                low = float(candles[-1].get("low", low))
        else:
            # Fallback: append current ticker data (less accurate)
            if current_price > 0:
                self.price_history.append(current_price)
                self.high_history.append(high)
                self.low_history.append(low)
                self.volume_history.append(volume)

                # Keep limited history
                max_len = max(self.roc_period, self.macd_slow, self.rsi_period) * 3
                if len(self.price_history) > max_len:
                    self.price_history = self.price_history[-max_len:]
                    self.high_history = self.high_history[-max_len:]
                    self.low_history = self.low_history[-max_len:]
                    self.volume_history = self.volume_history[-max_len:]

        # Check if we have enough data
        min_required = max(self.roc_period, self.macd_slow) + 5
        if len(self.price_history) < min_required:
            return {
                "signal": None,
                "reasoning": f"Insufficient candle data: {len(self.price_history)}/{min_required} periods"
            }

        # Generate signals
        signal = self._generate_signal(current_price, regime)

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_STRATEGY_GENERATION,
            model="momentum_v1",
            input_data={
                "price": current_price,
                "high": high,
                "low": low,
                "volume": volume,
                "regime": regime,
            },
            output_data={
                "signal_direction": signal.direction if signal else None,
                "momentum_type": signal.momentum_type.value if signal else None,
                "momentum_score": signal.momentum_score if signal else 0,
                "stop_price": signal.stop_price if signal else None,
                "position_size_pct": signal.position_size_pct if signal else 0,
            },
            explanation=signal.reasoning if signal else "No momentum signal",
        )

        if signal and signal.direction != "none":
            return {
                "signal": {
                    "strategy": "momentum",
                    "direction": signal.direction,
                    "momentum_type": signal.momentum_type.value,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "position_size_pct": signal.position_size_pct,
                    "momentum_score": signal.momentum_score,
                    "timeframe": "1h",  # Momentum uses 1H candles
                },
                "reasoning": signal.reasoning,
                "confidence": min(1.0, signal.momentum_score / 70),  # Convert score to confidence
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
    ) -> Optional[MomentumSignal]:
        """Generate momentum signal.

        Checks multiple momentum indicators and returns combined signal.

        Args:
            current_price: Current price
            regime: Market regime

        Returns:
            MomentumSignal
        """
        prices = np.array(self.price_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)
        volumes = np.array(self.volume_history)

        # Calculate indicators
        roc = self._calculate_roc(prices)
        rsi = self._calculate_rsi(prices)
        macd, macd_signal, macd_hist = self._calculate_macd(prices)
        atr = self._calculate_atr(highs, lows, prices)
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        current_volume = volumes[-1] if len(volumes) > 0 else 0

        # Calculate composite momentum score (0-100)
        momentum_score = self._calculate_momentum_score(roc, rsi, macd_hist, current_volume, avg_volume)

        signals = []

        # 1. ROC Breakout
        roc_signal = self._check_roc_breakout(current_price, roc, atr, momentum_score)
        if roc_signal:
            signals.append(roc_signal)

        # 2. RSI Momentum
        rsi_signal = self._check_rsi_momentum(current_price, rsi, prices, atr, momentum_score)
        if rsi_signal:
            signals.append(rsi_signal)

        # 3. Volume Surge
        volume_signal = self._check_volume_surge(
            current_price, current_volume, avg_volume, prices, atr, momentum_score
        )
        if volume_signal:
            signals.append(volume_signal)

        # 4. MACD Crossover
        macd_signal_result = self._check_macd_crossover(
            current_price, macd, macd_signal, macd_hist, atr, momentum_score
        )
        if macd_signal_result:
            signals.append(macd_signal_result)

        # Return strongest signal (by momentum score)
        if signals:
            return max(signals, key=lambda s: s.momentum_score)

        return MomentumSignal(
            direction="none",
            momentum_type=MomentumType.ROC_BREAKOUT,
            entry_price=current_price,
            stop_price=0,
            target_price=None,
            position_size_pct=0,
            momentum_score=momentum_score,
            reasoning=f"No momentum signal. Score: {momentum_score:.0f}, ROC: {roc*100:.1f}%, RSI: {rsi:.1f}"
        )

    def _check_roc_breakout(
        self,
        current_price: float,
        roc: float,
        atr: float,
        momentum_score: float,
    ) -> Optional[MomentumSignal]:
        """Check for Rate of Change breakout.

        Args:
            current_price: Current price
            roc: Rate of change
            atr: Average True Range
            momentum_score: Composite momentum score

        Returns:
            MomentumSignal if ROC breakout detected
        """
        # Bullish: ROC > threshold
        if roc > self.roc_threshold:
            stop_price = current_price - (self.atr_multiplier * atr)
            target_price = current_price + (3 * atr)  # 3:1 R:R

            return MomentumSignal(
                direction="long",
                momentum_type=MomentumType.ROC_BREAKOUT,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * min(1.0, momentum_score / 70),
                momentum_score=momentum_score,
                reasoning=f"ROC Breakout LONG: ROC {roc*100:.1f}% > {self.roc_threshold*100:.0f}% threshold. Momentum score: {momentum_score:.0f}"
            )

        # Bearish: ROC < -threshold
        if roc < -self.roc_threshold:
            stop_price = current_price + (self.atr_multiplier * atr)
            target_price = current_price - (3 * atr)

            return MomentumSignal(
                direction="short",
                momentum_type=MomentumType.ROC_BREAKOUT,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.8 * min(1.0, momentum_score / 70),
                momentum_score=momentum_score,
                reasoning=f"ROC Breakdown SHORT: ROC {roc*100:.1f}% < -{self.roc_threshold*100:.0f}% threshold. Momentum score: {momentum_score:.0f}"
            )

        return None

    def _check_rsi_momentum(
        self,
        current_price: float,
        rsi: float,
        prices: np.ndarray,
        atr: float,
        momentum_score: float,
    ) -> Optional[MomentumSignal]:
        """Check for RSI momentum signal.

        Args:
            current_price: Current price
            rsi: RSI value
            prices: Price history
            atr: Average True Range
            momentum_score: Composite momentum score

        Returns:
            MomentumSignal if RSI momentum detected
        """
        # Bullish: RSI > 60 and price making higher highs
        if rsi > 60 and current_price > np.max(prices[-5:-1]):
            stop_price = current_price - (self.atr_multiplier * atr)
            target_price = current_price + (2.5 * atr)

            return MomentumSignal(
                direction="long",
                momentum_type=MomentumType.RSI_MOMENTUM,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.7,
                momentum_score=momentum_score,
                reasoning=f"RSI Momentum LONG: RSI {rsi:.1f} > 60 with price breakout. New high above recent range."
            )

        # Bearish: RSI < 40 and price making lower lows
        if rsi < 40 and current_price < np.min(prices[-5:-1]):
            stop_price = current_price + (self.atr_multiplier * atr)
            target_price = current_price - (2.5 * atr)

            return MomentumSignal(
                direction="short",
                momentum_type=MomentumType.RSI_MOMENTUM,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.6,
                momentum_score=momentum_score,
                reasoning=f"RSI Momentum SHORT: RSI {rsi:.1f} < 40 with price breakdown. New low below recent range."
            )

        return None

    def _check_volume_surge(
        self,
        current_price: float,
        current_volume: float,
        avg_volume: float,
        prices: np.ndarray,
        atr: float,
        momentum_score: float,
    ) -> Optional[MomentumSignal]:
        """Check for volume surge signal.

        Args:
            current_price: Current price
            current_volume: Current volume
            avg_volume: Average volume
            prices: Price history
            atr: Average True Range
            momentum_score: Composite momentum score

        Returns:
            MomentumSignal if volume surge detected
        """
        if avg_volume <= 0:
            return None

        volume_ratio = current_volume / avg_volume

        # Volume surge threshold met
        if volume_ratio < self.volume_mult:
            return None

        # Determine direction from price movement
        price_change = (current_price - prices[-5]) / prices[-5] if prices[-5] > 0 else 0

        # Bullish: Volume surge with price rise
        if price_change > 0.01:  # > 1% move up
            stop_price = current_price - (self.atr_multiplier * atr)
            target_price = current_price + (2 * atr)

            return MomentumSignal(
                direction="long",
                momentum_type=MomentumType.VOLUME_SURGE,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.8,
                momentum_score=momentum_score,
                reasoning=f"Volume Surge LONG: {volume_ratio:.1f}x avg volume with {price_change*100:.1f}% price rise. Strong buying conviction."
            )

        # Bearish: Volume surge with price drop
        if price_change < -0.01:  # > 1% move down
            stop_price = current_price + (self.atr_multiplier * atr)
            target_price = current_price - (2 * atr)

            return MomentumSignal(
                direction="short",
                momentum_type=MomentumType.VOLUME_SURGE,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.7,
                momentum_score=momentum_score,
                reasoning=f"Volume Surge SHORT: {volume_ratio:.1f}x avg volume with {price_change*100:.1f}% price drop. Strong selling pressure."
            )

        return None

    def _check_macd_crossover(
        self,
        current_price: float,
        macd: float,
        macd_signal: float,
        macd_hist: float,
        atr: float,
        momentum_score: float,
    ) -> Optional[MomentumSignal]:
        """Check for MACD crossover signal.

        Args:
            current_price: Current price
            macd: MACD line
            macd_signal: MACD signal line
            macd_hist: MACD histogram
            atr: Average True Range
            momentum_score: Composite momentum score

        Returns:
            MomentumSignal if MACD crossover detected
        """
        # Bullish: MACD > Signal and histogram expanding
        if macd > macd_signal and macd_hist > 0:
            stop_price = current_price - (self.atr_multiplier * atr)
            target_price = current_price + (2.5 * atr)

            return MomentumSignal(
                direction="long",
                momentum_type=MomentumType.MACD_CROSSOVER,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.6,
                momentum_score=momentum_score,
                reasoning=f"MACD Crossover LONG: MACD {macd:.4f} > Signal {macd_signal:.4f}, Histogram: {macd_hist:.4f}"
            )

        # Bearish: MACD < Signal and histogram expanding negative
        if macd < macd_signal and macd_hist < 0:
            stop_price = current_price + (self.atr_multiplier * atr)
            target_price = current_price - (2.5 * atr)

            return MomentumSignal(
                direction="short",
                momentum_type=MomentumType.MACD_CROSSOVER,
                entry_price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                position_size_pct=self.max_position_pct * 0.5,
                momentum_score=momentum_score,
                reasoning=f"MACD Crossover SHORT: MACD {macd:.4f} < Signal {macd_signal:.4f}, Histogram: {macd_hist:.4f}"
            )

        return None

    def _calculate_momentum_score(
        self,
        roc: float,
        rsi: float,
        macd_hist: float,
        current_volume: float,
        avg_volume: float,
    ) -> float:
        """Calculate composite momentum score (0-100).

        Args:
            roc: Rate of change
            rsi: RSI value
            macd_hist: MACD histogram
            current_volume: Current volume
            avg_volume: Average volume

        Returns:
            Momentum score 0-100
        """
        score = 50  # Neutral base

        # ROC contribution (+/- 20 points)
        roc_score = min(20, max(-20, roc * 200))
        score += roc_score

        # RSI contribution (+/- 15 points)
        if rsi > 60:
            score += min(15, (rsi - 50) * 0.5)
        elif rsi < 40:
            score -= min(15, (50 - rsi) * 0.5)

        # MACD histogram contribution (+/- 10 points)
        macd_score = min(10, max(-10, macd_hist * 1000))
        score += macd_score

        # Volume contribution (+/- 5 points)
        if avg_volume > 0:
            vol_ratio = current_volume / avg_volume
            vol_score = min(5, max(-5, (vol_ratio - 1) * 5))
            score += vol_score

        return max(0, min(100, score))

    def _calculate_roc(self, prices: np.ndarray) -> float:
        """Calculate Rate of Change."""
        if len(prices) < self.roc_period + 1:
            return 0.0

        past_price = prices[-self.roc_period - 1]
        current_price = prices[-1]

        if past_price == 0:
            return 0.0

        return (current_price - past_price) / past_price

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(prices[-self.rsi_period - 1:])
        gains = deltas.copy()
        losses = deltas.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> tuple:
        """Calculate MACD, Signal, and Histogram."""
        if len(prices) < self.macd_slow + self.macd_signal:
            return 0.0, 0.0, 0.0

        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)

        macd = ema_fast - ema_slow

        # For signal line, we need MACD history
        # Simplified: use single value
        signal = macd * 0.9  # Approximate

        histogram = macd - signal

        return macd, signal, histogram

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
