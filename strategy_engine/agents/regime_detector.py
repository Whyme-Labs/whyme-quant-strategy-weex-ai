"""Regime Detector Agent for market regime classification.

Based on research from @web3tinkle's 36-regime classification and @hackertrader's
Two Strategies framework, this agent classifies market conditions to determine
whether Mean Reversion or Trend Following strategies should be applied.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_MARKET_ANALYSIS


class VolatilityRegime(Enum):
    """Volatility classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendRegime(Enum):
    """Trend classification."""
    STRONG_DOWN = "strong_down"
    WEAK_DOWN = "weak_down"
    SIDEWAYS = "sideways"
    WEAK_UP = "weak_up"
    STRONG_UP = "strong_up"


class VolumeRegime(Enum):
    """Volume classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class StrategyType(Enum):
    """Recommended strategy type based on regime."""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    NEUTRAL = "neutral"  # No clear edge


@dataclass
class MarketRegime:
    """Complete market regime classification."""
    volatility: VolatilityRegime
    trend: TrendRegime
    volume: VolumeRegime
    recommended_strategy: StrategyType
    confidence: float
    reasoning: str


class RegimeDetectorAgent(BaseAgent):
    """Agent that detects market regime and recommends strategy type.

    Uses multiple indicators to classify:
    - Volatility: ATR, Bollinger Band width
    - Trend: EMA slopes, price position relative to EMAs
    - Volume: Relative volume compared to average

    Maps regime to either Mean Reversion or Trend Following strategy.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize regime detector.

        Args:
            config: Configuration dictionary with:
                - ema_periods: List of EMA periods [8, 20, 50]
                - atr_period: ATR calculation period (default 14)
                - lookback_period: Historical data lookback (default 50)
        """
        super().__init__(config)
        self.ema_periods = config.get("ema_periods", [8, 20, 50])
        self.atr_period = config.get("atr_period", 14)
        self.lookback_period = config.get("lookback_period", 50)
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and determine regime.

        Args:
            context: Dictionary containing:
                - market_data: Current market data
                - historical_prices: Optional price history

        Returns:
            Dictionary with regime classification and strategy recommendation
        """
        market_data = context.get("market_data", {})

        # Update price history
        current_price = market_data.get("price", 0)
        current_volume = market_data.get("volume", 0)
        current_high = market_data.get("high_24h", current_price)
        current_low = market_data.get("low_24h", current_price)

        if current_price > 0:
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            self.high_history.append(current_high)
            self.low_history.append(current_low)

            # Keep only lookback period
            max_history = self.lookback_period * 2
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
                self.volume_history = self.volume_history[-max_history:]
                self.high_history = self.high_history[-max_history:]
                self.low_history = self.low_history[-max_history:]

        # Need minimum data for analysis
        if len(self.price_history) < 20:
            regime = MarketRegime(
                volatility=VolatilityRegime.MEDIUM,
                trend=TrendRegime.SIDEWAYS,
                volume=VolumeRegime.NORMAL,
                recommended_strategy=StrategyType.NEUTRAL,
                confidence=0.3,
                reasoning="Insufficient data for regime detection. Collecting more samples."
            )
        else:
            regime = self._detect_regime()

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_MARKET_ANALYSIS,
            model="regime_detector_v1",
            input_data={
                "price": current_price,
                "volume": current_volume,
                "price_history_length": len(self.price_history),
            },
            output_data={
                "volatility": regime.volatility.value,
                "trend": regime.trend.value,
                "volume": regime.volume.value,
                "recommended_strategy": regime.recommended_strategy.value,
                "confidence": regime.confidence,
            },
            explanation=regime.reasoning,
        )

        return {
            "regime": {
                "volatility": regime.volatility.value,
                "trend": regime.trend.value,
                "volume": regime.volume.value,
            },
            "recommended_strategy": regime.recommended_strategy.value,
            "confidence": regime.confidence,
            "reasoning": regime.reasoning,
        }

    def _detect_regime(self) -> MarketRegime:
        """Detect current market regime using technical analysis.

        Returns:
            MarketRegime with full classification
        """
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)

        # Calculate volatility regime
        volatility_regime, vol_score = self._classify_volatility(prices)

        # Calculate trend regime
        trend_regime, trend_score = self._classify_trend(prices)

        # Calculate volume regime
        volume_regime, vol_regime_score = self._classify_volume(volumes)

        # Determine recommended strategy based on regime combination
        strategy, confidence, reasoning = self._map_regime_to_strategy(
            volatility_regime, trend_regime, volume_regime,
            vol_score, trend_score
        )

        return MarketRegime(
            volatility=volatility_regime,
            trend=trend_regime,
            volume=volume_regime,
            recommended_strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _classify_volatility(self, prices: np.ndarray) -> tuple[VolatilityRegime, float]:
        """Classify volatility using ATR-like measure.

        Args:
            prices: Array of historical prices

        Returns:
            Tuple of (VolatilityRegime, normalized_score)
        """
        if len(prices) < 2:
            return VolatilityRegime.MEDIUM, 0.5

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Use standard deviation of returns as volatility proxy
        volatility = np.std(returns) * 100  # As percentage

        # Historical average volatility thresholds (can be calibrated)
        if volatility < 1.0:  # < 1% daily moves
            return VolatilityRegime.LOW, volatility / 3.0
        elif volatility < 3.0:  # 1-3% daily moves
            return VolatilityRegime.MEDIUM, 0.5
        else:  # > 3% daily moves
            return VolatilityRegime.HIGH, min(volatility / 5.0, 1.0)

    def _classify_trend(self, prices: np.ndarray) -> tuple[TrendRegime, float]:
        """Classify trend using EMA analysis.

        Uses EMA(8), EMA(20), EMA(50) alignment and slopes.

        Args:
            prices: Array of historical prices

        Returns:
            Tuple of (TrendRegime, trend_strength_score)
        """
        if len(prices) < max(self.ema_periods):
            return TrendRegime.SIDEWAYS, 0.0

        # Calculate EMAs
        ema_8 = self._calculate_ema(prices, 8)
        ema_20 = self._calculate_ema(prices, 20)
        ema_50 = self._calculate_ema(prices, min(50, len(prices)))

        current_price = prices[-1]

        # Check EMA alignment
        bullish_alignment = ema_8 > ema_20 > ema_50
        bearish_alignment = ema_8 < ema_20 < ema_50

        # Calculate EMA slopes (recent change)
        if len(prices) > 5:
            ema_8_slope = (ema_8 - self._calculate_ema(prices[:-5], 8)) / ema_8 * 100
        else:
            ema_8_slope = 0

        # Price position relative to EMAs
        above_all = current_price > ema_8 > ema_20
        below_all = current_price < ema_8 < ema_20

        # Determine trend
        if bullish_alignment and above_all and ema_8_slope > 0.5:
            return TrendRegime.STRONG_UP, min(abs(ema_8_slope) / 2, 1.0)
        elif bullish_alignment or (above_all and ema_8_slope > 0):
            return TrendRegime.WEAK_UP, 0.5
        elif bearish_alignment and below_all and ema_8_slope < -0.5:
            return TrendRegime.STRONG_DOWN, min(abs(ema_8_slope) / 2, 1.0)
        elif bearish_alignment or (below_all and ema_8_slope < 0):
            return TrendRegime.WEAK_DOWN, 0.5
        else:
            return TrendRegime.SIDEWAYS, 0.0

    def _classify_volume(self, volumes: np.ndarray) -> tuple[VolumeRegime, float]:
        """Classify volume relative to average.

        Args:
            volumes: Array of historical volumes

        Returns:
            Tuple of (VolumeRegime, relative_volume_score)
        """
        if len(volumes) < 10:
            return VolumeRegime.NORMAL, 0.5

        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]

        if avg_volume == 0:
            return VolumeRegime.NORMAL, 0.5

        relative_volume = current_volume / avg_volume

        if relative_volume < 0.7:
            return VolumeRegime.LOW, relative_volume
        elif relative_volume > 1.5:
            return VolumeRegime.HIGH, min(relative_volume / 2, 1.0)
        else:
            return VolumeRegime.NORMAL, 0.5

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average.

        Args:
            prices: Price array
            period: EMA period

        Returns:
            Current EMA value
        """
        if len(prices) < period:
            return np.mean(prices)

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _map_regime_to_strategy(
        self,
        volatility: VolatilityRegime,
        trend: TrendRegime,
        volume: VolumeRegime,
        vol_score: float,
        trend_score: float,
    ) -> tuple[StrategyType, float, str]:
        """Map regime to recommended strategy.

        Based on @hackertrader's insight:
        - Trend Following works in trending markets
        - Mean Reversion works in ranging/extreme markets

        Args:
            volatility: Volatility regime
            trend: Trend regime
            volume: Volume regime
            vol_score: Volatility intensity score
            trend_score: Trend strength score

        Returns:
            Tuple of (StrategyType, confidence, reasoning)
        """
        reasoning_parts = []

        # Strong trends favor trend following
        if trend in [TrendRegime.STRONG_UP, TrendRegime.STRONG_DOWN]:
            if volume == VolumeRegime.HIGH:
                # Strong trend with volume confirmation
                confidence = 0.8 + trend_score * 0.2
                reasoning_parts.append(
                    f"Strong {trend.value} trend with high volume confirms momentum"
                )
                return StrategyType.TREND_FOLLOWING, confidence, ". ".join(reasoning_parts)
            else:
                confidence = 0.6 + trend_score * 0.2
                reasoning_parts.append(
                    f"Strong {trend.value} trend detected, but volume not confirming"
                )
                return StrategyType.TREND_FOLLOWING, confidence, ". ".join(reasoning_parts)

        # High volatility + sideways = mean reversion opportunity
        if volatility == VolatilityRegime.HIGH and trend == TrendRegime.SIDEWAYS:
            confidence = 0.7 + vol_score * 0.2
            reasoning_parts.append(
                "High volatility in sideways market - extremes likely to revert"
            )
            return StrategyType.MEAN_REVERSION, confidence, ". ".join(reasoning_parts)

        # Low volatility often precedes breakouts (VCP pattern)
        if volatility == VolatilityRegime.LOW:
            if trend in [TrendRegime.WEAK_UP, TrendRegime.WEAK_DOWN]:
                reasoning_parts.append(
                    "Low volatility with weak trend - potential VCP breakout setup"
                )
                return StrategyType.TREND_FOLLOWING, 0.6, ". ".join(reasoning_parts)
            else:
                reasoning_parts.append(
                    "Low volatility sideways - waiting for direction"
                )
                return StrategyType.NEUTRAL, 0.4, ". ".join(reasoning_parts)

        # Weak trends with normal volatility
        if trend in [TrendRegime.WEAK_UP, TrendRegime.WEAK_DOWN]:
            if volume == VolumeRegime.HIGH:
                # Potential reversal or continuation
                reasoning_parts.append(
                    "Weak trend with high volume - watching for breakout or reversal"
                )
                return StrategyType.NEUTRAL, 0.5, ". ".join(reasoning_parts)
            else:
                # Favor trend continuation
                reasoning_parts.append(
                    f"Weak {trend.value} trend continuing - light trend following"
                )
                return StrategyType.TREND_FOLLOWING, 0.55, ". ".join(reasoning_parts)

        # Default: sideways with normal volatility
        reasoning_parts.append(
            "No clear regime edge - reducing exposure"
        )
        return StrategyType.NEUTRAL, 0.4, ". ".join(reasoning_parts)
