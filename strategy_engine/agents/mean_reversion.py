"""Mean Reversion Strategy Agent.

Based on research insights:
- Jason Shapiro (34%/year): "Fades positioning + failed news, not price extremes"
- High win rate, low risk/reward ratio
- "Size is the last line of defense"
- P&L Profile: Many small wins, few large losses

IMPORTANT: RSI and Bollinger Bands are noisy on small timeframes.
This agent uses higher timeframe confirmation (4H/Daily equivalent) to filter signals.

Now uses centralized IndicatorsService for consistent indicator calculations.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
import pandas as pd

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_STRATEGY_GENERATION

if TYPE_CHECKING:
    from ..services.indicators_service import IndicatorsService
    from ..services.market_data_service import MarketDataService


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

    Uses IndicatorsService for:
    - Bollinger Bands (bb_upper, bb_mid, bb_lower, bb_pct)
    - RSI (rsi)
    - Stochastic (stoch_k, stoch_d) for confirmation
    - ATR for stop loss calculation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        indicators_service: Optional["IndicatorsService"] = None,
        market_data_service: Optional["MarketDataService"] = None,
    ):
        """Initialize mean reversion agent.

        Args:
            config: Configuration with:
                - rsi_oversold: RSI oversold threshold (default 30)
                - rsi_overbought: RSI overbought threshold (default 70)
                - max_position_pct: Maximum position as % of account (default 0.05)
                - target_return_pct: Target return per trade (default 0.02)
                - stop_loss_pct: Stop loss percentage (default 0.02)
                - timeframe: Candle timeframe (default "4h")
            indicators_service: Centralized indicator service
            market_data_service: Market data service for candles
        """
        super().__init__(config)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.max_position_pct = config.get("max_position_pct", 0.05)
        self.target_return_pct = config.get("target_return_pct", 0.02)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)
        self.timeframe = config.get("timeframe", "4h")

        # Service dependencies
        self.indicators_service = indicators_service
        self.market_data_service = market_data_service

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate mean reversion signals.

        Uses IndicatorsService for indicator calculations.

        Args:
            context: Dictionary containing:
                - market_data: Current market data
                - regime: Current market regime

        Returns:
            Dictionary with signal details
        """
        market_data = context.get("market_data", {})
        regime = context.get("regime", {})
        symbol = market_data.get("symbol", "BTCUSDT")
        current_price = market_data.get("price", 0)

        # Get indicators from service or calculate from candle data
        indicators = await self._get_indicators(symbol, market_data)

        if not indicators:
            return {
                "signal": None,
                "reasoning": f"Unable to calculate indicators for {self.timeframe} timeframe"
            }

        # Check for required indicators
        required = ["rsi", "bb_upper", "bb_mid", "bb_lower"]
        missing = [ind for ind in required if ind not in indicators or indicators[ind] is None]
        if missing:
            return {
                "signal": None,
                "reasoning": f"Missing required indicators: {missing}"
            }

        # Generate signal
        signal = self._generate_signal(current_price, indicators, regime)

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_STRATEGY_GENERATION,
            model="mean_reversion_v2",
            input_data={
                "price": current_price,
                "rsi": indicators.get("rsi"),
                "bb_pct": indicators.get("bb_pct"),
                "stoch_k": indicators.get("stoch_k"),
                "regime": regime,
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
                    "timeframe": self.timeframe,
                },
                "reasoning": signal.reasoning,
                "confidence": 0.7 if signal.strength == SignalStrength.STRONG else 0.5,
            }
        else:
            return {
                "signal": None,
                "reasoning": signal.reasoning if signal else "No signal"
            }

    async def _get_indicators(
        self,
        symbol: str,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get indicators from IndicatorsService or calculate from candles.

        Args:
            symbol: Trading symbol
            market_data: Market data with candles

        Returns:
            Dictionary of indicator values
        """
        # Method 1: Use IndicatorsService if available
        if self.indicators_service and self.market_data_service:
            try:
                indicators = await self.indicators_service.calculate_indicators(
                    symbol, self.timeframe, limit=100
                )
                if indicators:
                    return indicators
            except Exception:
                pass

        # Method 2: Calculate from candle data in context
        candle_key = f"candles_{self.timeframe}"
        candles = market_data.get(candle_key) or market_data.get("candles_4h") or []

        if not candles or len(candles) < 20:
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Ensure required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                return {}

        # Calculate indicators using IndicatorsService if available
        if self.indicators_service:
            df = self.indicators_service.calculate_all(df)
        else:
            # Fallback: manual calculation
            df = self._calculate_indicators_fallback(df)

        if df.empty:
            return {}

        # Return latest values
        latest = df.iloc[-1]
        return {
            "rsi": latest.get("rsi"),
            "bb_upper": latest.get("bb_upper"),
            "bb_mid": latest.get("bb_mid"),
            "bb_lower": latest.get("bb_lower"),
            "bb_pct": latest.get("bb_pct"),
            "stoch_k": latest.get("stoch_k"),
            "stoch_d": latest.get("stoch_d"),
            "atr": latest.get("atr"),
            "close": latest.get("close"),
        }

    def _calculate_indicators_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback indicator calculation when IndicatorsService unavailable.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with indicators added
        """
        close = df["close"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["bb_upper"] = sma + (std * 2)
        df["bb_mid"] = sma
        df["bb_lower"] = sma - (std * 2)
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        high = df["high"]
        low = df["low"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        return df

    def _generate_signal(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> Optional[MeanReversionSignal]:
        """Generate mean reversion trading signal.

        Args:
            current_price: Current market price
            indicators: Pre-calculated indicators
            regime: Current market regime

        Returns:
            MeanReversionSignal or None
        """
        rsi = indicators.get("rsi", 50)
        bb_upper = indicators.get("bb_upper", current_price)
        bb_mid = indicators.get("bb_mid", current_price)
        bb_lower = indicators.get("bb_lower", current_price)
        bb_pct = indicators.get("bb_pct", 0.5)
        stoch_k = indicators.get("stoch_k")
        atr = indicators.get("atr", current_price * 0.02)

        # Calculate deviation from mean
        deviation_pct = (current_price - bb_mid) / bb_mid * 100 if bb_mid > 0 else 0

        reasoning_parts = []

        # Check for oversold conditions (long signal)
        oversold_bb = current_price < bb_lower or bb_pct < 0
        oversold_rsi = rsi < self.rsi_oversold
        oversold_stoch = stoch_k is not None and stoch_k < 20

        # Check for overbought conditions (short signal)
        overbought_bb = current_price > bb_upper or bb_pct > 1
        overbought_rsi = rsi > self.rsi_overbought
        overbought_stoch = stoch_k is not None and stoch_k > 80

        # Count confirmations
        long_confirmations = sum([oversold_bb, oversold_rsi, oversold_stoch])
        short_confirmations = sum([overbought_bb, overbought_rsi, overbought_stoch])

        # Calculate stop loss using ATR
        atr_stop_mult = 2.0
        stop_distance = atr * atr_stop_mult if atr else current_price * self.stop_loss_pct

        # Determine signal
        if long_confirmations >= 2:
            # Strong or moderate long signal
            if oversold_bb:
                reasoning_parts.append(f"Price below lower BB (${current_price:,.2f} < ${bb_lower:,.2f})")
            if oversold_rsi:
                reasoning_parts.append(f"RSI oversold ({rsi:.1f} < {self.rsi_oversold})")
            if oversold_stoch:
                reasoning_parts.append(f"Stochastic oversold ({stoch_k:.1f})")

            strength = SignalStrength.STRONG if long_confirmations >= 3 else SignalStrength.MODERATE
            position_mult = 0.8 if strength == SignalStrength.STRONG else 0.5

            signal = MeanReversionSignal(
                direction="long",
                strength=strength,
                entry_price=current_price,
                target_price=bb_mid,
                stop_price=current_price - stop_distance,
                position_size_pct=self.max_position_pct * position_mult,
                reasoning=". ".join(reasoning_parts) + f". {strength.value.title()} mean reversion long setup."
            )
            return signal

        elif short_confirmations >= 2:
            # Strong or moderate short signal
            if overbought_bb:
                reasoning_parts.append(f"Price above upper BB (${current_price:,.2f} > ${bb_upper:,.2f})")
            if overbought_rsi:
                reasoning_parts.append(f"RSI overbought ({rsi:.1f} > {self.rsi_overbought})")
            if overbought_stoch:
                reasoning_parts.append(f"Stochastic overbought ({stoch_k:.1f})")

            strength = SignalStrength.STRONG if short_confirmations >= 3 else SignalStrength.MODERATE
            position_mult = 0.6 if strength == SignalStrength.STRONG else 0.3  # More conservative for shorts

            signal = MeanReversionSignal(
                direction="short",
                strength=strength,
                entry_price=current_price,
                target_price=bb_mid,
                stop_price=current_price + stop_distance,
                position_size_pct=self.max_position_pct * position_mult,
                reasoning=". ".join(reasoning_parts) + f". {strength.value.title()} mean reversion short setup."
            )
            return signal

        else:
            # No signal
            return MeanReversionSignal(
                direction="none",
                strength=SignalStrength.NONE,
                entry_price=current_price,
                target_price=bb_mid,
                stop_price=None,
                position_size_pct=0,
                reasoning=f"Price within normal range. RSI: {rsi:.1f}, BB%: {bb_pct:.2f}, Deviation: {deviation_pct:.1f}%"
            )
