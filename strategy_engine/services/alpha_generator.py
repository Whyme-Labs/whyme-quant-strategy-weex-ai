"""Alpha Generator Service - Signal Aggregation and Scoring.

Combines multiple signals from:
- Trend indicators (EMA, SuperTrend, ADX)
- Momentum indicators (RSI, MACD, Stochastic)
- Volume indicators (OBV, CMF, MFI)
- Multi-timeframe confirmation
- Chart pattern signals

Produces a unified alpha score for trade decisions.
Alpha = "finding edge" = combining weak signals into strong signals.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from .market_data_service import MarketDataService
    from .indicators_service import IndicatorsService
    from .pattern_detector import PatternDetector


@dataclass
class AlphaSignal:
    """Aggregated alpha signal."""
    symbol: str
    alpha: float  # -1 to +1
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0 to 1
    components: Dict[str, float]  # Score breakdown
    reasoning: str
    timeframe_alignment: Dict[str, str]
    active_patterns: List[str]


class AlphaGenerator:
    """Generates alpha signals by aggregating multiple signal sources.

    Alpha generation is the core of quantitative trading - finding
    edge by combining multiple weak signals into stronger signals.
    """

    # Default weights for signal components
    DEFAULT_WEIGHTS = {
        "trend": 0.30,      # Trend indicators (EMA, ADX, SuperTrend)
        "momentum": 0.25,   # Momentum (RSI, MACD, Stochastic)
        "volume": 0.15,     # Volume (OBV, CMF, MFI)
        "mtf": 0.20,        # Multi-timeframe confirmation
        "patterns": 0.10,   # Chart patterns
    }

    # Thresholds for direction classification
    LONG_THRESHOLD = 0.3
    SHORT_THRESHOLD = -0.3

    def __init__(
        self,
        market_data_service: Optional["MarketDataService"] = None,
        indicators_service: Optional["IndicatorsService"] = None,
        pattern_detector: Optional["PatternDetector"] = None,
        weights: Optional[Dict[str, float]] = None,
        timeframes: Optional[List[str]] = None,
    ):
        """Initialize Alpha Generator.

        Args:
            market_data_service: Service for fetching candle data
            indicators_service: Service for indicator calculations
            pattern_detector: Service for pattern detection
            weights: Custom weights for signal components
            timeframes: Timeframes for MTF analysis (default: ["1h", "4h", "1d"])
        """
        self.market_data_service = market_data_service
        self.indicators_service = indicators_service
        self.pattern_detector = pattern_detector
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.timeframes = timeframes or ["1h", "4h", "1d"]

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    async def generate_alpha(self, symbol: str) -> AlphaSignal:
        """Generate alpha signal for a symbol.

        Aggregates signals from all sources with weighted scoring.

        Args:
            symbol: Trading pair

        Returns:
            AlphaSignal with direction, confidence, and breakdown
        """
        components = {}
        reasoning_parts = []
        timeframe_alignment = {}
        active_patterns = []

        # Get multi-timeframe data
        mtf_data = await self._get_mtf_data(symbol)

        if not mtf_data:
            return AlphaSignal(
                symbol=symbol,
                alpha=0.0,
                direction="NEUTRAL",
                confidence=0.0,
                components={},
                reasoning="No market data available",
                timeframe_alignment={},
                active_patterns=[],
            )

        # Calculate each component score

        # 1. Trend Score
        trend_score, trend_reason = self._score_trend(mtf_data)
        components["trend"] = trend_score
        reasoning_parts.append(f"Trend: {trend_reason}")

        # 2. Momentum Score
        momentum_score, momentum_reason = self._score_momentum(mtf_data)
        components["momentum"] = momentum_score
        reasoning_parts.append(f"Momentum: {momentum_reason}")

        # 3. Volume Score
        volume_score, volume_reason = self._score_volume(mtf_data)
        components["volume"] = volume_score
        reasoning_parts.append(f"Volume: {volume_reason}")

        # 4. Multi-Timeframe Confirmation
        mtf_score, mtf_reason, tf_alignment = self._score_mtf_confirmation(mtf_data)
        components["mtf"] = mtf_score
        timeframe_alignment = tf_alignment
        reasoning_parts.append(f"MTF: {mtf_reason}")

        # 5. Pattern Score
        pattern_score, pattern_reason, patterns = await self._score_patterns(symbol, mtf_data)
        components["patterns"] = pattern_score
        active_patterns = patterns
        reasoning_parts.append(f"Patterns: {pattern_reason}")

        # Calculate weighted alpha
        alpha = sum(
            score * self.weights.get(component, 0)
            for component, score in components.items()
        )

        # Clip to [-1, 1]
        alpha = np.clip(alpha, -1, 1)

        # Determine direction
        if alpha > self.LONG_THRESHOLD:
            direction = "LONG"
        elif alpha < self.SHORT_THRESHOLD:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Confidence is the absolute alpha value
        confidence = abs(alpha)

        # Build reasoning
        reasoning = " | ".join(reasoning_parts)

        return AlphaSignal(
            symbol=symbol,
            alpha=alpha,
            direction=direction,
            confidence=confidence,
            components=components,
            reasoning=reasoning,
            timeframe_alignment=timeframe_alignment,
            active_patterns=active_patterns,
        )

    async def _get_mtf_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get multi-timeframe data with indicators calculated.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary of timeframe -> DataFrame with indicators
        """
        if not self.market_data_service:
            return {}

        mtf_data = {}

        for tf in self.timeframes:
            try:
                df = await self.market_data_service.get_candles(symbol, tf, limit=100)

                if df.empty:
                    continue

                # Calculate indicators
                if self.indicators_service:
                    df = self.indicators_service.calculate_all(df)

                mtf_data[tf] = df

            except Exception as e:
                logger.warning(f"Failed to get {tf} data for {symbol}: {e}")

        return mtf_data

    def _score_trend(
        self,
        mtf_data: Dict[str, pd.DataFrame],
    ) -> tuple[float, str]:
        """Score trend indicators.

        Considers:
        - EMA alignment (8 > 20 > 50 = bullish)
        - ADX trend strength
        - SuperTrend direction

        Returns:
            (score, reasoning)
        """
        scores = []
        reasons = []

        # Use primary timeframe (first available)
        primary_tf = next((tf for tf in self.timeframes if tf in mtf_data), None)
        if not primary_tf:
            return 0.0, "No data"

        df = mtf_data[primary_tf]
        latest = df.iloc[-1]

        # EMA Alignment
        if all(k in latest for k in ["ema_8", "ema_20", "ema_50"]):
            ema_8 = latest["ema_8"]
            ema_20 = latest["ema_20"]
            ema_50 = latest["ema_50"]

            if ema_8 > ema_20 > ema_50:
                scores.append(1.0)
                reasons.append("EMA bullish")
            elif ema_8 < ema_20 < ema_50:
                scores.append(-1.0)
                reasons.append("EMA bearish")
            else:
                scores.append(0.0)
                reasons.append("EMA mixed")

        # ADX Trend Strength
        if "adx" in latest and "di_plus" in latest and "di_minus" in latest:
            adx = latest["adx"]
            di_plus = latest["di_plus"]
            di_minus = latest["di_minus"]

            if adx > 25:  # Strong trend
                direction = 0.5 if di_plus > di_minus else -0.5
                scores.append(direction)
                reasons.append(f"ADX strong ({adx:.0f})")
            else:
                reasons.append(f"ADX weak ({adx:.0f})")

        # SuperTrend
        if "supertrend_dir" in latest:
            st_dir = latest["supertrend_dir"]
            scores.append(1.0 if st_dir == 1 else -1.0)
            reasons.append("ST bullish" if st_dir == 1 else "ST bearish")

        # MACD
        if "macd_hist" in latest:
            hist = latest["macd_hist"]
            macd_score = np.clip(hist / (abs(hist) + 0.01), -1, 1) if hist != 0 else 0
            scores.append(macd_score * 0.5)  # Lower weight
            reasons.append(f"MACD {'↑' if hist > 0 else '↓'}")

        if not scores:
            return 0.0, "No trend indicators"

        return np.mean(scores), ", ".join(reasons[:3])

    def _score_momentum(
        self,
        mtf_data: Dict[str, pd.DataFrame],
    ) -> tuple[float, str]:
        """Score momentum indicators.

        Considers:
        - RSI zones (oversold/overbought)
        - MACD crossovers
        - Stochastic %K/%D

        Returns:
            (score, reasoning)
        """
        scores = []
        reasons = []

        primary_tf = next((tf for tf in self.timeframes if tf in mtf_data), None)
        if not primary_tf:
            return 0.0, "No data"

        df = mtf_data[primary_tf]
        latest = df.iloc[-1]

        # RSI
        if "rsi" in latest:
            rsi = latest["rsi"]
            if rsi < 30:
                scores.append(0.8)  # Oversold = bullish opportunity
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                scores.append(-0.8)  # Overbought = bearish opportunity
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi > 50:
                scores.append(0.3)
                reasons.append(f"RSI bullish ({rsi:.0f})")
            else:
                scores.append(-0.3)
                reasons.append(f"RSI bearish ({rsi:.0f})")

        # Stochastic
        if "stoch_k" in latest and "stoch_d" in latest:
            k = latest["stoch_k"]
            d = latest["stoch_d"]

            if k < 20 and k > d:  # Oversold with bullish cross
                scores.append(0.7)
                reasons.append("Stoch oversold bullish")
            elif k > 80 and k < d:  # Overbought with bearish cross
                scores.append(-0.7)
                reasons.append("Stoch overbought bearish")
            elif k > d:
                scores.append(0.2)
            else:
                scores.append(-0.2)

        # CCI
        if "cci" in latest:
            cci = latest["cci"]
            cci_score = np.clip(cci / 200, -1, 1)
            scores.append(cci_score * 0.5)

        # Williams %R
        if "willr" in latest:
            willr = latest["willr"]
            if willr < -80:
                scores.append(0.5)
                reasons.append("Williams oversold")
            elif willr > -20:
                scores.append(-0.5)
                reasons.append("Williams overbought")

        if not scores:
            return 0.0, "No momentum indicators"

        return np.mean(scores), ", ".join(reasons[:2])

    def _score_volume(
        self,
        mtf_data: Dict[str, pd.DataFrame],
    ) -> tuple[float, str]:
        """Score volume indicators.

        Considers:
        - OBV trend
        - CMF (Chaikin Money Flow)
        - MFI (Money Flow Index)

        Returns:
            (score, reasoning)
        """
        scores = []
        reasons = []

        primary_tf = next((tf for tf in self.timeframes if tf in mtf_data), None)
        if not primary_tf:
            return 0.0, "No data"

        df = mtf_data[primary_tf]
        latest = df.iloc[-1]

        # CMF
        if "cmf" in latest:
            cmf = latest["cmf"]
            cmf_score = np.clip(cmf * 5, -1, 1)
            scores.append(cmf_score)
            if cmf > 0.1:
                reasons.append("CMF accumulation")
            elif cmf < -0.1:
                reasons.append("CMF distribution")

        # MFI
        if "mfi" in latest:
            mfi = latest["mfi"]
            mfi_score = (mfi - 50) / 50
            scores.append(mfi_score * 0.7)
            if mfi < 20:
                reasons.append("MFI oversold")
            elif mfi > 80:
                reasons.append("MFI overbought")

        # OBV trend (compare recent to average)
        if "obv" in latest and len(df) > 10:
            obv = latest["obv"]
            obv_sma = df["obv"].rolling(10).mean().iloc[-1]

            if obv_sma != 0:
                obv_ratio = (obv - obv_sma) / abs(obv_sma)
                scores.append(np.clip(obv_ratio, -1, 1))

                if obv_ratio > 0.05:
                    reasons.append("OBV rising")
                elif obv_ratio < -0.05:
                    reasons.append("OBV falling")

        if not scores:
            return 0.0, "No volume indicators"

        return np.mean(scores), ", ".join(reasons[:2]) if reasons else "Volume neutral"

    def _score_mtf_confirmation(
        self,
        mtf_data: Dict[str, pd.DataFrame],
    ) -> tuple[float, str, Dict[str, str]]:
        """Score multi-timeframe trend alignment.

        Best signals occur when multiple timeframes align:
        - 1H bullish + 4H bullish + 1D bullish = strong long

        Returns:
            (score, reasoning, timeframe_alignment)
        """
        tf_signals = {}
        alignment = {}

        for tf, df in mtf_data.items():
            if df.empty:
                continue

            latest = df.iloc[-1]

            # Determine trend on this timeframe
            bullish_signals = 0
            bearish_signals = 0

            # EMA check
            if "ema_8" in latest and "ema_20" in latest:
                if latest["ema_8"] > latest["ema_20"]:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # Price vs SMA20
            if "sma_20" in latest and "close" in latest:
                if latest["close"] > latest["sma_20"]:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # SuperTrend
            if "supertrend_dir" in latest:
                if latest["supertrend_dir"] == 1:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # Classify
            if bullish_signals > bearish_signals:
                tf_signals[tf] = 1
                alignment[tf] = "BULLISH"
            elif bearish_signals > bullish_signals:
                tf_signals[tf] = -1
                alignment[tf] = "BEARISH"
            else:
                tf_signals[tf] = 0
                alignment[tf] = "NEUTRAL"

        if not tf_signals:
            return 0.0, "No MTF data", {}

        # Score based on alignment
        signals = list(tf_signals.values())

        # All aligned in same direction = high score
        if all(s > 0 for s in signals):
            return 1.0, "All TF bullish", alignment
        elif all(s < 0 for s in signals):
            return -1.0, "All TF bearish", alignment
        elif all(s == 0 for s in signals):
            return 0.0, "All TF neutral", alignment
        else:
            # Weighted by timeframe importance (longer TF = more weight)
            weights = {"1h": 0.2, "4h": 0.3, "1d": 0.5}
            weighted_score = sum(
                tf_signals.get(tf, 0) * weights.get(tf, 0.3)
                for tf in tf_signals
            )
            return weighted_score, "Mixed TF", alignment

    async def _score_patterns(
        self,
        symbol: str,
        mtf_data: Dict[str, pd.DataFrame],
    ) -> tuple[float, str, List[str]]:
        """Score chart patterns.

        Returns:
            (score, reasoning, active_patterns)
        """
        if not self.pattern_detector:
            return 0.0, "No pattern detector", []

        primary_tf = next((tf for tf in self.timeframes if tf in mtf_data), None)
        if not primary_tf:
            return 0.0, "No data", []

        df = mtf_data[primary_tf]
        patterns = self.pattern_detector.detect_all(df)

        if not patterns:
            return 0.0, "No patterns", []

        # Aggregate pattern signals
        bullish_patterns = []
        bearish_patterns = []

        from .pattern_detector import PatternSignal

        for p in patterns[:5]:  # Top 5 patterns
            if p.signal == PatternSignal.BULLISH:
                bullish_patterns.append(f"{p.pattern_type.value} ({p.confidence:.0%})")
            elif p.signal == PatternSignal.BEARISH:
                bearish_patterns.append(f"{p.pattern_type.value} ({p.confidence:.0%})")

        # Score based on pattern count and confidence
        bullish_score = sum(p.confidence for p in patterns if p.signal == PatternSignal.BULLISH)
        bearish_score = sum(p.confidence for p in patterns if p.signal == PatternSignal.BEARISH)

        net_score = (bullish_score - bearish_score) / max(bullish_score + bearish_score, 1)
        net_score = np.clip(net_score, -1, 1)

        all_patterns = bullish_patterns + bearish_patterns

        if bullish_patterns and not bearish_patterns:
            reason = f"Bullish: {bullish_patterns[0]}"
        elif bearish_patterns and not bullish_patterns:
            reason = f"Bearish: {bearish_patterns[0]}"
        elif bullish_patterns and bearish_patterns:
            reason = "Mixed patterns"
        else:
            reason = "No clear patterns"

        return net_score, reason, all_patterns

    def get_alpha_summary(self, alpha_signal: AlphaSignal) -> Dict[str, Any]:
        """Get formatted summary of alpha signal.

        Args:
            alpha_signal: Generated alpha signal

        Returns:
            Formatted summary for display/logging
        """
        return {
            "symbol": alpha_signal.symbol,
            "direction": alpha_signal.direction,
            "alpha": f"{alpha_signal.alpha:+.2f}",
            "confidence": f"{alpha_signal.confidence:.0%}",
            "components": {
                k: f"{v:+.2f}" for k, v in alpha_signal.components.items()
            },
            "timeframe_alignment": alpha_signal.timeframe_alignment,
            "patterns": alpha_signal.active_patterns[:3],
            "reasoning": alpha_signal.reasoning,
        }
