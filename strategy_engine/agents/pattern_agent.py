"""Pattern-Based Strategy Agent.

Uses the PatternDetector service to identify classical chart patterns:
- Head and Shoulders / Inverse Head and Shoulders
- Double Top / Double Bottom
- Ascending/Descending/Symmetric Triangles
- Bull/Bear Flags
- Rising/Falling Wedges

Generates trading signals when patterns are detected and confirmed.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_STRATEGY_GENERATION

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService
    from ..services.pattern_detector import PatternDetector, Pattern, PatternSignal


@dataclass
class PatternTradeSignal:
    """Pattern-based trading signal."""
    direction: str  # "long", "short", or "none"
    pattern_name: str
    pattern_type: str
    entry_price: float
    stop_price: float
    target_price: float
    position_size_pct: float
    confidence: float
    reasoning: str


class PatternAgent(BaseAgent):
    """Pattern-Based Strategy Agent.

    Strategy Philosophy:
    - Classical chart patterns are self-fulfilling prophecies
    - Institutional traders watch and act on these patterns
    - Patterns provide clear entry, stop, and target levels
    - Combine pattern signals with price action confirmation

    Patterns Detected:
    - Head and Shoulders (bearish reversal)
    - Inverse Head and Shoulders (bullish reversal)
    - Double Top / Double Bottom
    - Ascending Triangle (bullish continuation)
    - Descending Triangle (bearish continuation)
    - Symmetric Triangle (neutral until breakout)
    - Bull/Bear Flags (continuation)
    - Rising Wedge (bearish) / Falling Wedge (bullish)

    Entry Rules:
    - Pattern must be fully formed or activated (neckline broken)
    - Confidence > 0.65 for trade consideration
    - Price action confirms pattern direction

    Exit Rules:
    - Target: Measured move from pattern
    - Stop: Pattern invalidation level
    """

    def __init__(
        self,
        config: Dict[str, Any],
        market_data_service: Optional["MarketDataService"] = None,
        pattern_detector: Optional["PatternDetector"] = None,
    ):
        """Initialize pattern agent.

        Args:
            config: Configuration with:
                - min_confidence: Minimum pattern confidence (default 0.65)
                - max_position_pct: Maximum position size (default 0.06)
                - timeframe: Timeframe for pattern detection (default "4h")
            market_data_service: Service for fetching candle data
            pattern_detector: Service for pattern detection
        """
        super().__init__(config)
        self.min_confidence = config.get("min_confidence", 0.65)
        self.max_position_pct = config.get("max_position_pct", 0.06)
        self.timeframe = config.get("timeframe", "4h")

        self.market_data_service = market_data_service
        self.pattern_detector = pattern_detector

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate pattern-based signals.

        Args:
            context: Dictionary containing:
                - market_data: Current market data with OHLCV candles
                - regime: Current market regime

        Returns:
            Dictionary with signal details
        """
        market_data = context.get("market_data", {})
        regime = context.get("regime", {})
        symbol = market_data.get("symbol", "BTCUSDT")

        current_price = market_data.get("price", 0)

        # Get candle data for pattern detection
        candles_df = await self._get_candle_data(symbol, market_data)

        if candles_df is None or candles_df.empty:
            return {
                "signal": None,
                "reasoning": "No candle data available for pattern detection"
            }

        if len(candles_df) < 30:
            return {
                "signal": None,
                "reasoning": f"Insufficient data: {len(candles_df)}/30 candles required"
            }

        # Detect patterns
        patterns = self._detect_patterns(candles_df)

        if not patterns:
            return {
                "signal": None,
                "reasoning": "No chart patterns detected"
            }

        # Generate signal from best pattern
        signal = self._generate_signal(patterns, current_price, regime)

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_STRATEGY_GENERATION,
            model="pattern_v1",
            input_data={
                "price": current_price,
                "patterns_detected": len(patterns),
                "top_pattern": patterns[0].pattern_type.value if patterns else None,
                "regime": regime,
            },
            output_data={
                "signal_direction": signal.direction if signal else None,
                "pattern_name": signal.pattern_name if signal else None,
                "confidence": signal.confidence if signal else 0,
                "target_price": signal.target_price if signal else None,
                "stop_price": signal.stop_price if signal else None,
            },
            explanation=signal.reasoning if signal else "No pattern signal",
        )

        if signal and signal.direction != "none":
            return {
                "signal": {
                    "strategy": "pattern",
                    "direction": signal.direction,
                    "pattern_name": signal.pattern_name,
                    "pattern_type": signal.pattern_type,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "position_size_pct": signal.position_size_pct,
                    "timeframe": self.timeframe,
                },
                "reasoning": signal.reasoning,
                "confidence": signal.confidence,
            }
        else:
            return {
                "signal": None,
                "reasoning": signal.reasoning if signal else "No pattern signal"
            }

    async def _get_candle_data(
        self,
        symbol: str,
        market_data: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """Get candle data for pattern detection.

        Args:
            symbol: Trading pair
            market_data: Current market data (may contain candles)

        Returns:
            DataFrame with OHLCV data
        """
        # Try to get from market data service first
        if self.market_data_service:
            try:
                df = await self.market_data_service.get_candles(
                    symbol, self.timeframe, limit=100
                )
                if not df.empty:
                    return df
            except Exception:
                pass

        # Fallback: use candles from market_data context
        candle_key = f"candles_{self.timeframe}"
        candles = market_data.get(candle_key) or market_data.get("candles_4h") or []

        if not candles:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return None

        return df

    def _detect_patterns(self, df: pd.DataFrame) -> List["Pattern"]:
        """Detect chart patterns in the data.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of detected patterns sorted by confidence
        """
        if not self.pattern_detector:
            # Fallback: create pattern detector if not provided
            from ..services.pattern_detector import PatternDetector
            self.pattern_detector = PatternDetector(
                min_pattern_bars=20,
                peak_distance=5,
                peak_prominence=0.01,
            )

        patterns = self.pattern_detector.detect_all(df)

        # Filter by minimum confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]

        return patterns

    def _generate_signal(
        self,
        patterns: List["Pattern"],
        current_price: float,
        regime: Dict[str, Any],
    ) -> PatternTradeSignal:
        """Generate trading signal from detected patterns.

        Args:
            patterns: List of detected patterns
            current_price: Current market price
            regime: Market regime

        Returns:
            PatternTradeSignal
        """
        from ..services.pattern_detector import PatternSignal

        if not patterns:
            return PatternTradeSignal(
                direction="none",
                pattern_name="none",
                pattern_type="none",
                entry_price=current_price,
                stop_price=0,
                target_price=0,
                position_size_pct=0,
                confidence=0,
                reasoning="No patterns meet minimum confidence threshold"
            )

        # Take the highest confidence pattern
        best_pattern = patterns[0]

        # Determine direction
        if best_pattern.signal == PatternSignal.BULLISH:
            direction = "long"
        elif best_pattern.signal == PatternSignal.BEARISH:
            direction = "short"
        else:
            # Neutral pattern - wait for breakout
            return PatternTradeSignal(
                direction="none",
                pattern_name=best_pattern.pattern_type.value,
                pattern_type=best_pattern.pattern_type.value,
                entry_price=current_price,
                stop_price=0,
                target_price=0,
                position_size_pct=0,
                confidence=best_pattern.confidence,
                reasoning=f"{best_pattern.description}. Waiting for breakout direction."
            )

        # Get target and stop from pattern
        target_price = best_pattern.target_price or current_price
        stop_price = best_pattern.stop_price or current_price

        # Calculate position size based on confidence
        base_size = self.max_position_pct
        confidence_factor = min(1.0, best_pattern.confidence / 0.8)  # Scale up to 80% confidence
        position_size = base_size * confidence_factor

        # Reduce size for reversal patterns (more risky)
        reversal_patterns = [
            "head_and_shoulders", "inverse_head_and_shoulders",
            "double_top", "double_bottom",
            "rising_wedge", "falling_wedge"
        ]
        if best_pattern.pattern_type.value in reversal_patterns:
            position_size *= 0.7  # 30% reduction for reversal patterns

        # Build reasoning
        key_levels_str = ", ".join(
            f"{k}: ${v:,.2f}" for k, v in best_pattern.key_levels.items()
        )
        reasoning = (
            f"{best_pattern.description} | "
            f"Key levels: {key_levels_str} | "
            f"Confidence: {best_pattern.confidence:.0%}"
        )

        return PatternTradeSignal(
            direction=direction,
            pattern_name=best_pattern.pattern_type.value.replace("_", " ").title(),
            pattern_type=best_pattern.pattern_type.value,
            entry_price=current_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size_pct=position_size,
            confidence=best_pattern.confidence,
            reasoning=reasoning,
        )

    def get_detected_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of all detected patterns.

        Args:
            df: OHLCV DataFrame

        Returns:
            Pattern summary dictionary
        """
        patterns = self._detect_patterns(df)

        if not self.pattern_detector:
            return {"count": 0, "patterns": []}

        summary = self.pattern_detector.get_pattern_summary(patterns)

        return {
            **summary,
            "all_patterns": [
                {
                    "type": p.pattern_type.value,
                    "signal": p.signal.value,
                    "confidence": p.confidence,
                    "description": p.description,
                }
                for p in patterns[:5]
            ]
        }
