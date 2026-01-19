"""Mean Reversion Strategy Agent.

Based on research insights:
- Jason Shapiro (34%/year): "Fades positioning + failed news, not price extremes"
- High win rate, low risk/reward ratio
- "Size is the last line of defense"
- P&L Profile: Many small wins, few large losses

IMPORTANT: RSI and Bollinger Bands are noisy on small timeframes.
This agent uses higher timeframe confirmation (4H/Daily equivalent) to filter signals.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_STRATEGY_GENERATION


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal."""
    direction: str  # "long" or "short"
    strength: SignalStrength
    entry_price: float
    target_price: float
    stop_price: Optional[float]  # Can be flexible in MR
    position_size_pct: float  # Percentage of max position
    reasoning: str


class MeanReversionAgent(BaseAgent):
    """Mean Reversion Strategy Agent.

    Strategy Philosophy:
    - Buy low, sell high
    - Extremes don't last
    - High win rate, small gains per trade
    - Position sizing is the primary risk control

    Entry Signals:
    - Bollinger Band extremes (price outside bands)
    - RSI oversold/overbought
    - Failed breakout/breakdown
    - Excessive price deviation from moving average

    Exit Rules:
    - First green candle (for shorts) / first red candle (for longs)
    - Return to mean (20 EMA)
    - Fixed percentage bounce (e.g., 5%)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize mean reversion agent.

        Args:
            config: Configuration with:
                - bb_period: Bollinger Band period (default 20)
                - bb_std: Bollinger Band standard deviations (default 2)
                - rsi_period: RSI period (default 14)
                - rsi_oversold: RSI oversold threshold (default 30)
                - rsi_overbought: RSI overbought threshold (default 70)
                - max_position_pct: Maximum position as % of account (default 0.05)
                - target_return_pct: Target return per trade (default 0.02)
                - htf_multiplier: Higher timeframe multiplier for noise filtering (default 4)
                - min_htf_periods: Minimum higher timeframe periods needed (default 20)
        """
        super().__init__(config)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2.0)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.max_position_pct = config.get("max_position_pct", 0.05)
        self.target_return_pct = config.get("target_return_pct", 0.02)  # 2% target

        # Multi-timeframe settings (reduces noise)
        # htf_multiplier=4 means if base is 1H, HTF is 4H
        self.htf_multiplier = config.get("htf_multiplier", 4)
        self.min_htf_periods = config.get("min_htf_periods", 20)

        self.price_history: List[float] = []
        self.htf_price_history: List[float] = []  # Higher timeframe prices
        self._htf_counter = 0  # Counter for HTF aggregation

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate mean reversion signals.

        Uses multi-timeframe analysis:
        - Lower timeframe: Entry timing
        - Higher timeframe: Signal confirmation (reduces noise)

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

        # Use OHLCV candle data for proper calculations (prefer 4h for mean reversion)
        candles = market_data.get("candles_4h") or market_data.get("candles_1h") or []

        if candles and len(candles) >= self.min_htf_periods:
            # Extract close prices from candles for HTF analysis
            self.htf_price_history = [float(c.get("close", 0)) for c in candles]
            self.price_history = self.htf_price_history.copy()
        else:
            # Fallback: build history from ticks (less accurate)
            if current_price > 0:
                self.price_history.append(current_price)
                if len(self.price_history) > 200:
                    self.price_history = self.price_history[-200:]

                # Aggregate to higher timeframe (e.g., every 4 candles = 4H if base is 1H)
                self._htf_counter += 1
                if self._htf_counter >= self.htf_multiplier:
                    self.htf_price_history.append(current_price)
                    self._htf_counter = 0
                    if len(self.htf_price_history) > 100:
                        self.htf_price_history = self.htf_price_history[-100:]

        # Check if we have enough data (need HTF data for confirmation)
        if len(self.htf_price_history) < self.min_htf_periods:
            return {
                "signal": None,
                "reasoning": f"Building HTF data: {len(self.htf_price_history)}/{self.min_htf_periods} periods. Need {self.min_htf_periods} candles for RSI/BB."
            }

        # Generate signal with HTF confirmation
        signal = self._generate_signal(current_price, regime)

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_STRATEGY_GENERATION,
            model="mean_reversion_v1",
            input_data={
                "price": current_price,
                "regime": regime,
                "bb_period": self.bb_period,
                "rsi_period": self.rsi_period,
            },
            output_data={
                "signal_direction": signal.direction if signal else None,
                "signal_strength": signal.strength.value if signal else None,
                "position_size_pct": signal.position_size_pct if signal else 0,
            },
            explanation=signal.reasoning if signal else "No mean reversion signal",
        )

        if signal and signal.strength != SignalStrength.NONE:
            return {
                "signal": {
                    "strategy": "mean_reversion",
                    "direction": signal.direction,
                    "strength": signal.strength.value,
                    "entry_price": signal.entry_price,
                    "target_price": signal.target_price,
                    "stop_price": signal.stop_price,
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
    ) -> Optional[MeanReversionSignal]:
        """Generate mean reversion trading signal.

        Args:
            current_price: Current market price
            regime: Current market regime

        Returns:
            MeanReversionSignal or None
        """
        prices = np.array(self.price_history)

        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
        rsi = self._calculate_rsi(prices)

        # Calculate deviation from mean
        deviation_pct = (current_price - bb_middle) / bb_middle * 100

        reasoning_parts = []

        # Check for oversold conditions (long signal)
        oversold_bb = current_price < bb_lower
        oversold_rsi = rsi < self.rsi_oversold

        # Check for overbought conditions (short signal)
        overbought_bb = current_price > bb_upper
        overbought_rsi = rsi > self.rsi_overbought

        # Determine signal
        if oversold_bb and oversold_rsi:
            # Strong long signal
            reasoning_parts.append(
                f"Price below lower BB ({current_price:.2f} < {bb_lower:.2f})"
            )
            reasoning_parts.append(f"RSI oversold ({rsi:.1f} < {self.rsi_oversold})")

            signal = MeanReversionSignal(
                direction="long",
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                target_price=bb_middle,  # Target mean
                stop_price=None,  # MR uses size as defense
                position_size_pct=self.max_position_pct * 0.8,  # 80% of max
                reasoning=". ".join(reasoning_parts) + ". Strong mean reversion long setup."
            )
            return signal

        elif oversold_bb or oversold_rsi:
            # Moderate long signal
            if oversold_bb:
                reasoning_parts.append(f"Price at lower BB ({deviation_pct:.1f}% below mean)")
            if oversold_rsi:
                reasoning_parts.append(f"RSI approaching oversold ({rsi:.1f})")

            signal = MeanReversionSignal(
                direction="long",
                strength=SignalStrength.MODERATE,
                entry_price=current_price,
                target_price=bb_middle,
                stop_price=None,
                position_size_pct=self.max_position_pct * 0.5,  # 50% of max
                reasoning=". ".join(reasoning_parts) + ". Moderate mean reversion long."
            )
            return signal

        elif overbought_bb and overbought_rsi:
            # Strong short signal
            reasoning_parts.append(
                f"Price above upper BB ({current_price:.2f} > {bb_upper:.2f})"
            )
            reasoning_parts.append(f"RSI overbought ({rsi:.1f} > {self.rsi_overbought})")

            signal = MeanReversionSignal(
                direction="short",
                strength=SignalStrength.STRONG,
                entry_price=current_price,
                target_price=bb_middle,
                stop_price=None,  # Caution: shorts have unlimited risk
                position_size_pct=self.max_position_pct * 0.6,  # More conservative for shorts
                reasoning=". ".join(reasoning_parts) + ". Strong mean reversion short setup."
            )
            return signal

        elif overbought_bb or overbought_rsi:
            # Moderate short signal
            if overbought_bb:
                reasoning_parts.append(f"Price at upper BB ({deviation_pct:.1f}% above mean)")
            if overbought_rsi:
                reasoning_parts.append(f"RSI approaching overbought ({rsi:.1f})")

            signal = MeanReversionSignal(
                direction="short",
                strength=SignalStrength.MODERATE,
                entry_price=current_price,
                target_price=bb_middle,
                stop_price=None,
                position_size_pct=self.max_position_pct * 0.3,  # Conservative
                reasoning=". ".join(reasoning_parts) + ". Moderate mean reversion short."
            )
            return signal

        else:
            # No signal
            return MeanReversionSignal(
                direction="none",
                strength=SignalStrength.NONE,
                entry_price=current_price,
                target_price=bb_middle,
                stop_price=None,
                position_size_pct=0,
                reasoning=f"Price within normal range. RSI: {rsi:.1f}, Deviation: {deviation_pct:.1f}%"
            )

    def _calculate_bollinger_bands(
        self,
        prices: np.ndarray
    ) -> tuple[float, float, float]:
        """Calculate Bollinger Bands.

        Args:
            prices: Price array

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(prices) < self.bb_period:
            mean = np.mean(prices)
            std = np.std(prices)
        else:
            mean = np.mean(prices[-self.bb_period:])
            std = np.std(prices[-self.bb_period:])

        upper = mean + (self.bb_std * std)
        lower = mean - (self.bb_std * std)

        return upper, mean, lower

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate Relative Strength Index.

        Args:
            prices: Price array

        Returns:
            RSI value (0-100)
        """
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral

        # Calculate price changes
        deltas = np.diff(prices[-self.rsi_period - 1:])

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
