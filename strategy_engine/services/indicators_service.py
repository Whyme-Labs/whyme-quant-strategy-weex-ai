"""Centralized Technical Indicators Service using pandas-ta.

This service provides:
- 50+ technical indicators via pandas-ta
- Consistent calculation across all agents
- Multi-timeframe indicator aggregation
- Signal scoring for alpha generation
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
from loguru import logger

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas-ta not installed. Using fallback indicator calculations.")

if TYPE_CHECKING:
    from .market_data_service import MarketDataService


class IndicatorsService:
    """Centralized technical indicator calculations.

    Uses pandas-ta for production-grade indicator calculations.
    Provides both raw indicator values and scored signals.
    """

    # Default indicator parameters
    DEFAULT_PARAMS = {
        # Trend
        "ema_fast": 8,
        "ema_medium": 20,
        "ema_slow": 50,
        "ema_long": 200,
        "sma_period": 20,
        "adx_period": 14,
        "supertrend_period": 10,
        "supertrend_multiplier": 3.0,

        # Momentum
        "rsi_period": 14,
        "stoch_k": 14,
        "stoch_d": 3,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "cci_period": 20,
        "roc_period": 10,
        "williams_period": 14,

        # Volatility
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "kc_period": 20,
        "kc_multiplier": 1.5,
        "donchian_period": 20,

        # Volume
        "obv_signal": 10,
        "cmf_period": 20,
        "mfi_period": 14,
        "vwap_anchor": "D",
    }

    def __init__(
        self,
        market_data_service: Optional["MarketDataService"] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Indicators Service.

        Args:
            market_data_service: Market data service for fetching candles
            params: Custom indicator parameters (overrides defaults)
        """
        self.market_data_service = market_data_service
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        if not HAS_PANDAS_TA:
            logger.warning("Running without pandas-ta - limited indicators available")

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators and add as columns to DataFrame.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with indicator columns added
        """
        if df.empty:
            return df

        # Make a copy to avoid modifying original
        result = df.copy()

        # Ensure proper column names
        result = self._normalize_columns(result)

        if HAS_PANDAS_TA:
            result = self._calculate_with_pandas_ta(result)
        else:
            result = self._calculate_fallback(result)

        return result

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names for pandas-ta compatibility."""
        # Map common variations to standard names
        column_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Timestamp": "timestamp",
        }

        df = df.rename(columns=column_map)
        return df

    def _calculate_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using pandas-ta."""
        try:
            # === TREND INDICATORS ===

            # EMAs
            df["ema_8"] = ta.ema(df["close"], length=self.params["ema_fast"])
            df["ema_20"] = ta.ema(df["close"], length=self.params["ema_medium"])
            df["ema_50"] = ta.ema(df["close"], length=self.params["ema_slow"])
            df["ema_200"] = ta.ema(df["close"], length=self.params["ema_long"])

            # SMA
            df["sma_20"] = ta.sma(df["close"], length=self.params["sma_period"])

            # MACD
            try:
                macd = ta.macd(
                    df["close"],
                    fast=self.params["macd_fast"],
                    slow=self.params["macd_slow"],
                    signal=self.params["macd_signal"],
                )
                if macd is not None and isinstance(macd, pd.DataFrame) and not macd.empty:
                    df["macd"] = macd.iloc[:, 0]
                    df["macd_signal"] = macd.iloc[:, 2]
                    df["macd_hist"] = macd.iloc[:, 1]
            except Exception as e:
                logger.debug(f"MACD calculation failed: {e}")

            # ADX
            try:
                adx = ta.adx(
                    df["high"],
                    df["low"],
                    df["close"],
                    length=self.params["adx_period"],
                )
                if adx is not None and isinstance(adx, pd.DataFrame) and not adx.empty:
                    df["adx"] = adx.iloc[:, 0]
                    df["di_plus"] = adx.iloc[:, 1]
                    df["di_minus"] = adx.iloc[:, 2]
            except Exception as e:
                logger.debug(f"ADX calculation failed: {e}")

            # SuperTrend
            try:
                supertrend = ta.supertrend(
                    df["high"],
                    df["low"],
                    df["close"],
                    length=self.params["supertrend_period"],
                    multiplier=self.params["supertrend_multiplier"],
                )
                if supertrend is not None and isinstance(supertrend, pd.DataFrame) and not supertrend.empty:
                    df["supertrend"] = supertrend.iloc[:, 0]
                    df["supertrend_dir"] = supertrend.iloc[:, 1]  # 1 = bullish, -1 = bearish
            except Exception as e:
                logger.debug(f"SuperTrend calculation failed: {e}")

            # Ichimoku
            try:
                ichimoku = ta.ichimoku(
                    df["high"],
                    df["low"],
                    df["close"],
                )
                # ichimoku returns a tuple of DataFrames; safely extract
                if ichimoku is not None:
                    if isinstance(ichimoku, tuple) and len(ichimoku) > 0:
                        ichi_df = ichimoku[0]
                        if ichi_df is not None and not ichi_df.empty and len(ichi_df.columns) >= 4:
                            df["ichi_tenkan"] = ichi_df.iloc[:, 0]
                            df["ichi_kijun"] = ichi_df.iloc[:, 1]
                            df["ichi_senkou_a"] = ichi_df.iloc[:, 2]
                            df["ichi_senkou_b"] = ichi_df.iloc[:, 3]
                    elif isinstance(ichimoku, pd.DataFrame) and not ichimoku.empty:
                        # Handle case where ichimoku returns single DataFrame
                        if len(ichimoku.columns) >= 4:
                            df["ichi_tenkan"] = ichimoku.iloc[:, 0]
                            df["ichi_kijun"] = ichimoku.iloc[:, 1]
                            df["ichi_senkou_a"] = ichimoku.iloc[:, 2]
                            df["ichi_senkou_b"] = ichimoku.iloc[:, 3]
            except Exception as e:
                logger.debug(f"Ichimoku calculation failed: {e}")

            # === MOMENTUM INDICATORS ===

            # RSI
            df["rsi"] = ta.rsi(df["close"], length=self.params["rsi_period"])

            # Stochastic
            try:
                stoch = ta.stoch(
                    df["high"],
                    df["low"],
                    df["close"],
                    k=self.params["stoch_k"],
                    d=self.params["stoch_d"],
                )
                if stoch is not None and isinstance(stoch, pd.DataFrame) and not stoch.empty:
                    df["stoch_k"] = stoch.iloc[:, 0]
                    df["stoch_d"] = stoch.iloc[:, 1]
            except Exception as e:
                logger.debug(f"Stochastic calculation failed: {e}")

            # Williams %R
            df["willr"] = ta.willr(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["williams_period"],
            )

            # CCI
            df["cci"] = ta.cci(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["cci_period"],
            )

            # ROC (Rate of Change)
            df["roc"] = ta.roc(df["close"], length=self.params["roc_period"])

            # === VOLATILITY INDICATORS ===

            # Bollinger Bands
            try:
                bbands = ta.bbands(
                    df["close"],
                    length=self.params["bb_period"],
                    std=self.params["bb_std"],
                )
                if bbands is not None and isinstance(bbands, pd.DataFrame) and not bbands.empty:
                    df["bb_lower"] = bbands.iloc[:, 0]
                    df["bb_mid"] = bbands.iloc[:, 1]
                    df["bb_upper"] = bbands.iloc[:, 2]
                    df["bb_width"] = bbands.iloc[:, 3]
                    df["bb_pct"] = bbands.iloc[:, 4]  # %B
            except Exception as e:
                logger.debug(f"Bollinger Bands calculation failed: {e}")

            # ATR
            df["atr"] = ta.atr(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["atr_period"],
            )

            # Keltner Channel
            try:
                kc = ta.kc(
                    df["high"],
                    df["low"],
                    df["close"],
                    length=self.params["kc_period"],
                    scalar=self.params["kc_multiplier"],
                )
                if kc is not None and isinstance(kc, pd.DataFrame) and not kc.empty:
                    df["kc_lower"] = kc.iloc[:, 0]
                    df["kc_mid"] = kc.iloc[:, 1]
                    df["kc_upper"] = kc.iloc[:, 2]
            except Exception as e:
                logger.debug(f"Keltner Channel calculation failed: {e}")

            # Donchian Channel
            try:
                donchian = ta.donchian(
                    df["high"],
                    df["low"],
                    lower_length=self.params["donchian_period"],
                    upper_length=self.params["donchian_period"],
                )
                if donchian is not None and isinstance(donchian, pd.DataFrame) and not donchian.empty:
                    df["dc_lower"] = donchian.iloc[:, 0]
                    df["dc_mid"] = donchian.iloc[:, 1]
                    df["dc_upper"] = donchian.iloc[:, 2]
            except Exception as e:
                logger.debug(f"Donchian Channel calculation failed: {e}")

            # === VOLUME INDICATORS ===

            # OBV
            df["obv"] = ta.obv(df["close"], df["volume"])

            # CMF (Chaikin Money Flow)
            df["cmf"] = ta.cmf(
                df["high"],
                df["low"],
                df["close"],
                df["volume"],
                length=self.params["cmf_period"],
            )

            # MFI (Money Flow Index)
            df["mfi"] = ta.mfi(
                df["high"],
                df["low"],
                df["close"],
                df["volume"],
                length=self.params["mfi_period"],
            )

            # VWAP (if timestamp available)
            if "timestamp" in df.columns:
                try:
                    df["vwap"] = ta.vwap(
                        df["high"],
                        df["low"],
                        df["close"],
                        df["volume"],
                    )
                except Exception:
                    # VWAP requires datetime index
                    pass

            # === DONCHIAN CHANNELS (for Turtle system) ===
            # 20-day channel
            df["high_20d"] = df["high"].rolling(20).max()
            df["low_20d"] = df["low"].rolling(20).min()
            # 55-day channel (Turtle System 2)
            df["high_55d"] = df["high"].rolling(55).max()
            df["low_55d"] = df["low"].rolling(55).min()
            # 10-day channel (for exits)
            df["high_10d"] = df["high"].rolling(10).max()
            df["low_10d"] = df["low"].rolling(10).min()

            # === ATR RATIO (for VCP pattern) ===
            if "atr" in df.columns:
                atr_20 = df["atr"].rolling(20).mean()
                df["atr_ratio"] = df["atr"] / atr_20.where(atr_20 > 0, 1)

            # === CANDLESTICK PATTERNS ===
            df = self._calculate_candlestick_patterns(df)

            logger.debug(f"Calculated {len([c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return df

    def _calculate_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback indicator calculations without pandas-ta."""
        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            # EMAs
            df["ema_8"] = close.ewm(span=8, adjust=False).mean()
            df["ema_20"] = close.ewm(span=20, adjust=False).mean()
            df["ema_50"] = close.ewm(span=50, adjust=False).mean()
            df["ema_200"] = close.ewm(span=200, adjust=False).mean()

            # SMA
            df["sma_20"] = close.rolling(20).mean()

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
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
            df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()

            # Donchian Channel
            df["dc_upper"] = high.rolling(20).max()
            df["dc_lower"] = low.rolling(20).min()
            df["dc_mid"] = (df["dc_upper"] + df["dc_lower"]) / 2

            # OBV
            obv = [0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.append(obv[-1] + volume.iloc[i])
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.append(obv[-1] - volume.iloc[i])
                else:
                    obv.append(obv[-1])
            df["obv"] = obv

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

            logger.debug("Calculated fallback indicators (limited set)")

        except Exception as e:
            logger.error(f"Error in fallback calculations: {e}")

        return df

    def get_indicator(
        self,
        df: pd.DataFrame,
        name: str,
        **params,
    ) -> Optional[pd.Series]:
        """Get a specific indicator with custom parameters.

        Args:
            df: OHLCV DataFrame
            name: Indicator name (e.g., "rsi", "ema", "macd")
            **params: Custom parameters for the indicator

        Returns:
            Series with indicator values
        """
        if not HAS_PANDAS_TA:
            logger.warning(f"pandas-ta not available for {name}")
            return None

        df = self._normalize_columns(df)

        try:
            indicator_func = getattr(ta, name, None)
            if indicator_func is None:
                logger.error(f"Unknown indicator: {name}")
                return None

            # Determine required columns based on indicator
            if name in ["rsi", "ema", "sma", "roc"]:
                return indicator_func(df["close"], **params)
            elif name in ["atr", "adx", "stoch", "willr", "cci", "supertrend"]:
                return indicator_func(df["high"], df["low"], df["close"], **params)
            elif name in ["obv"]:
                return indicator_func(df["close"], df["volume"], **params)
            elif name in ["cmf", "mfi"]:
                return indicator_func(df["high"], df["low"], df["close"], df["volume"], **params)
            elif name in ["bbands", "kc", "donchian"]:
                result = indicator_func(df["high"], df["low"], df["close"], **params)
                return result  # Returns DataFrame with multiple columns
            else:
                # Try with close price as default
                return indicator_func(df["close"], **params)

        except Exception as e:
            logger.error(f"Error calculating {name}: {e}")
            return None

    async def get_multi_timeframe_signals(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get indicator signals across multiple timeframes.

        Useful for multi-timeframe confirmation where you want to
        see if signals align across different time horizons.

        Args:
            symbol: Trading pair
            timeframes: List of timeframes (default: ["1h", "4h", "1d"])

        Returns:
            Dictionary with timeframe -> signal summary
        """
        if not self.market_data_service:
            logger.error("Market data service not configured")
            return {}

        tfs = timeframes or ["1h", "4h", "1d"]
        signals = {}

        for tf in tfs:
            try:
                # Get candles
                df = await self.market_data_service.get_candles(symbol, tf, limit=100)
                if df.empty:
                    continue

                # Calculate indicators
                df = self.calculate_all(df)

                # Extract latest values
                latest = df.iloc[-1]
                signals[tf] = self._extract_signal_summary(latest)

            except Exception as e:
                logger.error(f"Error getting signals for {symbol} {tf}: {e}")

        return signals

    def _extract_signal_summary(self, row: pd.Series) -> Dict[str, Any]:
        """Extract a summary of signals from indicator row.

        Args:
            row: DataFrame row with indicator values

        Returns:
            Dictionary with signal summary
        """
        summary = {
            "price": row.get("close", 0),
        }

        # Trend signals
        if "ema_8" in row and "ema_20" in row:
            summary["ema_trend"] = "bullish" if row["ema_8"] > row["ema_20"] else "bearish"

        if "supertrend_dir" in row:
            summary["supertrend"] = "bullish" if row["supertrend_dir"] == 1 else "bearish"

        if "adx" in row:
            summary["adx"] = row["adx"]
            summary["trend_strength"] = "strong" if row["adx"] > 25 else "weak"

        # Momentum signals
        if "rsi" in row:
            summary["rsi"] = row["rsi"]
            if row["rsi"] < 30:
                summary["rsi_signal"] = "oversold"
            elif row["rsi"] > 70:
                summary["rsi_signal"] = "overbought"
            else:
                summary["rsi_signal"] = "neutral"

        if "macd_hist" in row:
            summary["macd_momentum"] = "bullish" if row["macd_hist"] > 0 else "bearish"

        # Volatility signals
        if "bb_pct" in row:
            summary["bb_position"] = row["bb_pct"]
            if row["bb_pct"] < 0:
                summary["bb_signal"] = "below_lower"
            elif row["bb_pct"] > 1:
                summary["bb_signal"] = "above_upper"
            else:
                summary["bb_signal"] = "within_bands"

        if "atr" in row:
            summary["atr"] = row["atr"]

        # Volume signals
        if "cmf" in row:
            summary["cmf"] = row["cmf"]
            summary["money_flow"] = "bullish" if row["cmf"] > 0 else "bearish"

        if "mfi" in row:
            summary["mfi"] = row["mfi"]

        return summary

    def score_trend(self, df: pd.DataFrame) -> float:
        """Score trend strength from -1 (bearish) to +1 (bullish).

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Trend score
        """
        if df.empty:
            return 0.0

        latest = df.iloc[-1]
        scores = []

        # EMA alignment
        if all(k in latest for k in ["ema_8", "ema_20", "ema_50"]):
            if latest["ema_8"] > latest["ema_20"] > latest["ema_50"]:
                scores.append(1.0)
            elif latest["ema_8"] < latest["ema_20"] < latest["ema_50"]:
                scores.append(-1.0)
            else:
                scores.append(0.0)

        # SuperTrend
        if "supertrend_dir" in latest:
            scores.append(1.0 if latest["supertrend_dir"] == 1 else -1.0)

        # MACD
        if "macd_hist" in latest:
            hist = latest["macd_hist"]
            scores.append(np.clip(hist / (abs(hist) + 0.001) if hist != 0 else 0, -1, 1))

        # ADX direction
        if "di_plus" in latest and "di_minus" in latest:
            if latest["di_plus"] > latest["di_minus"]:
                scores.append(0.5)
            else:
                scores.append(-0.5)

        return np.mean(scores) if scores else 0.0

    def score_momentum(self, df: pd.DataFrame) -> float:
        """Score momentum from -1 (oversold/bearish) to +1 (overbought/bullish).

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Momentum score
        """
        if df.empty:
            return 0.0

        latest = df.iloc[-1]
        scores = []

        # RSI (normalized to -1 to +1)
        if "rsi" in latest:
            rsi = latest["rsi"]
            # 30 -> -1, 50 -> 0, 70 -> 1
            scores.append((rsi - 50) / 20)

        # Stochastic
        if "stoch_k" in latest:
            stoch = latest["stoch_k"]
            scores.append((stoch - 50) / 50)

        # CCI (normalized)
        if "cci" in latest:
            cci = latest["cci"]
            scores.append(np.clip(cci / 100, -1, 1))

        # Williams %R
        if "willr" in latest:
            willr = latest["willr"]
            scores.append(-willr / 50 - 1)  # -100 to 0 -> -1 to 1

        return np.clip(np.mean(scores) if scores else 0.0, -1, 1)

    def score_volume(self, df: pd.DataFrame) -> float:
        """Score volume signals from -1 (distribution) to +1 (accumulation).

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Volume score
        """
        if df.empty:
            return 0.0

        latest = df.iloc[-1]
        scores = []

        # CMF
        if "cmf" in latest:
            scores.append(np.clip(latest["cmf"] * 5, -1, 1))

        # MFI (normalized)
        if "mfi" in latest:
            mfi = latest["mfi"]
            scores.append((mfi - 50) / 50)

        # OBV trend (compare to moving average)
        if "obv" in latest and len(df) > 10:
            obv_sma = df["obv"].rolling(10).mean().iloc[-1]
            if obv_sma != 0:
                obv_ratio = (latest["obv"] - obv_sma) / abs(obv_sma)
                scores.append(np.clip(obv_ratio, -1, 1))

        return np.clip(np.mean(scores) if scores else 0.0, -1, 1)

    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick patterns using pandas-ta.

        Detects 60+ candlestick patterns and adds them as columns.
        Pattern values: 100 = bullish, -100 = bearish, 0 = no pattern

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with candlestick pattern columns added
        """
        if not HAS_PANDAS_TA or df.empty:
            return df

        try:
            # === SINGLE CANDLE REVERSAL PATTERNS ===

            # Doji - indecision/potential reversal
            doji = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="doji")
            if doji is not None and not doji.empty:
                df["cdl_doji"] = doji.iloc[:, 0] if hasattr(doji, 'iloc') else doji

            # Hammer - bullish reversal at bottom
            hammer = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="hammer")
            if hammer is not None and not hammer.empty:
                df["cdl_hammer"] = hammer.iloc[:, 0] if hasattr(hammer, 'iloc') else hammer

            # Inverted Hammer - bullish reversal
            inv_hammer = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="invertedhammer")
            if inv_hammer is not None and not inv_hammer.empty:
                df["cdl_inverted_hammer"] = inv_hammer.iloc[:, 0] if hasattr(inv_hammer, 'iloc') else inv_hammer

            # Hanging Man - bearish reversal at top
            hanging = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="hangingman")
            if hanging is not None and not hanging.empty:
                df["cdl_hanging_man"] = hanging.iloc[:, 0] if hasattr(hanging, 'iloc') else hanging

            # Shooting Star - bearish reversal
            shooting = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="shootingstar")
            if shooting is not None and not shooting.empty:
                df["cdl_shooting_star"] = shooting.iloc[:, 0] if hasattr(shooting, 'iloc') else shooting

            # Spinning Top - indecision
            spinning = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="spinningtop")
            if spinning is not None and not spinning.empty:
                df["cdl_spinning_top"] = spinning.iloc[:, 0] if hasattr(spinning, 'iloc') else spinning

            # Marubozu - strong momentum
            marubozu = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="marubozu")
            if marubozu is not None and not marubozu.empty:
                df["cdl_marubozu"] = marubozu.iloc[:, 0] if hasattr(marubozu, 'iloc') else marubozu

            # === TWO CANDLE PATTERNS ===

            # Bullish Engulfing - strong bullish reversal
            bull_engulf = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="engulfing")
            if bull_engulf is not None and not bull_engulf.empty:
                df["cdl_engulfing"] = bull_engulf.iloc[:, 0] if hasattr(bull_engulf, 'iloc') else bull_engulf

            # Harami - reversal pattern
            harami = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="harami")
            if harami is not None and not harami.empty:
                df["cdl_harami"] = harami.iloc[:, 0] if hasattr(harami, 'iloc') else harami

            # Piercing Line - bullish reversal
            piercing = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="piercing")
            if piercing is not None and not piercing.empty:
                df["cdl_piercing"] = piercing.iloc[:, 0] if hasattr(piercing, 'iloc') else piercing

            # Dark Cloud Cover - bearish reversal
            dark_cloud = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="darkcloudcover")
            if dark_cloud is not None and not dark_cloud.empty:
                df["cdl_dark_cloud"] = dark_cloud.iloc[:, 0] if hasattr(dark_cloud, 'iloc') else dark_cloud

            # Tweezer Top/Bottom
            tweezer = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="tweezer")
            if tweezer is not None:
                # Note: Some versions may not have tweezer, handle gracefully
                pass

            # === THREE CANDLE PATTERNS ===

            # Morning Star - strong bullish reversal
            morning_star = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="morningstar")
            if morning_star is not None and not morning_star.empty:
                df["cdl_morning_star"] = morning_star.iloc[:, 0] if hasattr(morning_star, 'iloc') else morning_star

            # Evening Star - strong bearish reversal
            evening_star = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="eveningstar")
            if evening_star is not None and not evening_star.empty:
                df["cdl_evening_star"] = evening_star.iloc[:, 0] if hasattr(evening_star, 'iloc') else evening_star

            # Three White Soldiers - strong bullish
            soldiers = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="3whitesoldiers")
            if soldiers is not None and not soldiers.empty:
                df["cdl_three_white_soldiers"] = soldiers.iloc[:, 0] if hasattr(soldiers, 'iloc') else soldiers

            # Three Black Crows - strong bearish
            crows = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="3blackcrows")
            if crows is not None and not crows.empty:
                df["cdl_three_black_crows"] = crows.iloc[:, 0] if hasattr(crows, 'iloc') else crows

            # Three Inside Up/Down
            three_inside = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="3inside")
            if three_inside is not None and not three_inside.empty:
                df["cdl_three_inside"] = three_inside.iloc[:, 0] if hasattr(three_inside, 'iloc') else three_inside

            # Three Outside Up/Down
            three_outside = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="3outside")
            if three_outside is not None and not three_outside.empty:
                df["cdl_three_outside"] = three_outside.iloc[:, 0] if hasattr(three_outside, 'iloc') else three_outside

            # === ADDITIONAL PATTERNS ===

            # Dragonfly Doji - bullish reversal
            dragonfly = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="dragonflydoji")
            if dragonfly is not None and not dragonfly.empty:
                df["cdl_dragonfly_doji"] = dragonfly.iloc[:, 0] if hasattr(dragonfly, 'iloc') else dragonfly

            # Gravestone Doji - bearish reversal
            gravestone = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="gravestonedoji")
            if gravestone is not None and not gravestone.empty:
                df["cdl_gravestone_doji"] = gravestone.iloc[:, 0] if hasattr(gravestone, 'iloc') else gravestone

            # Long Legged Doji
            longlegged = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="longleggeddoji")
            if longlegged is not None and not longlegged.empty:
                df["cdl_longleg_doji"] = longlegged.iloc[:, 0] if hasattr(longlegged, 'iloc') else longlegged

            # Belt Hold
            belthold = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="belthold")
            if belthold is not None and not belthold.empty:
                df["cdl_belthold"] = belthold.iloc[:, 0] if hasattr(belthold, 'iloc') else belthold

            # Kicking pattern - very strong signal
            kicking = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="kicking")
            if kicking is not None and not kicking.empty:
                df["cdl_kicking"] = kicking.iloc[:, 0] if hasattr(kicking, 'iloc') else kicking

            # === AGGREGATE CANDLESTICK SIGNALS ===
            # Sum all bullish and bearish patterns for overall signal
            cdl_cols = [c for c in df.columns if c.startswith("cdl_")]
            if cdl_cols:
                df["cdl_bullish_count"] = df[cdl_cols].apply(
                    lambda row: sum(1 for v in row if pd.notna(v) and v > 0), axis=1
                )
                df["cdl_bearish_count"] = df[cdl_cols].apply(
                    lambda row: sum(1 for v in row if pd.notna(v) and v < 0), axis=1
                )
                df["cdl_net_signal"] = df["cdl_bullish_count"] - df["cdl_bearish_count"]

            logger.debug(f"Calculated {len(cdl_cols)} candlestick patterns")

        except Exception as e:
            logger.warning(f"Error calculating candlestick patterns: {e}")

        return df

    def get_candlestick_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of candlestick pattern signals.

        Args:
            df: DataFrame with candlestick patterns calculated

        Returns:
            Dictionary with pattern signals
        """
        if df.empty:
            return {"bullish": [], "bearish": [], "net_signal": 0}

        latest = df.iloc[-1]
        bullish_patterns = []
        bearish_patterns = []

        # Check each candlestick column
        cdl_cols = [c for c in df.columns if c.startswith("cdl_") and not c.endswith("_count") and c != "cdl_net_signal"]

        for col in cdl_cols:
            if col in latest and pd.notna(latest[col]):
                value = latest[col]
                pattern_name = col.replace("cdl_", "").replace("_", " ").title()
                if value > 0:
                    bullish_patterns.append(pattern_name)
                elif value < 0:
                    bearish_patterns.append(pattern_name)

        return {
            "bullish": bullish_patterns,
            "bearish": bearish_patterns,
            "bullish_count": len(bullish_patterns),
            "bearish_count": len(bearish_patterns),
            "net_signal": len(bullish_patterns) - len(bearish_patterns),
        }

    async def calculate_indicators(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Calculate indicators for a symbol/timeframe and return latest values.

        Args:
            symbol: Trading pair
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            limit: Number of candles to fetch

        Returns:
            Dictionary of indicator name -> latest value
        """
        if not self.market_data_service:
            logger.error("Market data service not configured")
            return {}

        try:
            # Get candles as list of dicts
            candles = await self.market_data_service.get_candles(symbol, timeframe, limit=limit)

            if not candles:
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df = self._normalize_columns(df)

            # Calculate all indicators
            df = self.calculate_all(df)

            if df.empty:
                return {}

            # Return latest row as dict
            latest = df.iloc[-1].to_dict()

            # Clean up NaN values
            return {k: (v if pd.notna(v) else None) for k, v in latest.items()}

        except Exception as e:
            # Use fallback indicators - this is handled gracefully
            logger.debug(f"pandas-ta indicators unavailable for {symbol} {timeframe}, using fallback: {e}")
            try:
                # Try fallback calculation
                df = pd.DataFrame(candles)
                df = self._normalize_columns(df)
                df = self._calculate_fallback(df)
                if not df.empty:
                    latest = df.iloc[-1].to_dict()
                    return {k: (v if pd.notna(v) else None) for k, v in latest.items()}
            except Exception as fallback_error:
                logger.warning(f"Fallback indicators also failed for {symbol} {timeframe}: {fallback_error}")
            return {}
