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
            macd = ta.macd(
                df["close"],
                fast=self.params["macd_fast"],
                slow=self.params["macd_slow"],
                signal=self.params["macd_signal"],
            )
            if macd is not None and not macd.empty:
                df["macd"] = macd.iloc[:, 0]
                df["macd_signal"] = macd.iloc[:, 2]
                df["macd_hist"] = macd.iloc[:, 1]

            # ADX
            adx = ta.adx(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["adx_period"],
            )
            if adx is not None and not adx.empty:
                df["adx"] = adx.iloc[:, 0]
                df["di_plus"] = adx.iloc[:, 1]
                df["di_minus"] = adx.iloc[:, 2]

            # SuperTrend
            supertrend = ta.supertrend(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["supertrend_period"],
                multiplier=self.params["supertrend_multiplier"],
            )
            if supertrend is not None and not supertrend.empty:
                df["supertrend"] = supertrend.iloc[:, 0]
                df["supertrend_dir"] = supertrend.iloc[:, 1]  # 1 = bullish, -1 = bearish

            # Ichimoku
            ichimoku = ta.ichimoku(
                df["high"],
                df["low"],
                df["close"],
            )
            if ichimoku is not None and len(ichimoku) > 0 and not ichimoku[0].empty:
                ichi_df = ichimoku[0]
                df["ichi_tenkan"] = ichi_df.iloc[:, 0]
                df["ichi_kijun"] = ichi_df.iloc[:, 1]
                df["ichi_senkou_a"] = ichi_df.iloc[:, 2]
                df["ichi_senkou_b"] = ichi_df.iloc[:, 3]

            # === MOMENTUM INDICATORS ===

            # RSI
            df["rsi"] = ta.rsi(df["close"], length=self.params["rsi_period"])

            # Stochastic
            stoch = ta.stoch(
                df["high"],
                df["low"],
                df["close"],
                k=self.params["stoch_k"],
                d=self.params["stoch_d"],
            )
            if stoch is not None and not stoch.empty:
                df["stoch_k"] = stoch.iloc[:, 0]
                df["stoch_d"] = stoch.iloc[:, 1]

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
            bbands = ta.bbands(
                df["close"],
                length=self.params["bb_period"],
                std=self.params["bb_std"],
            )
            if bbands is not None and not bbands.empty:
                df["bb_lower"] = bbands.iloc[:, 0]
                df["bb_mid"] = bbands.iloc[:, 1]
                df["bb_upper"] = bbands.iloc[:, 2]
                df["bb_width"] = bbands.iloc[:, 3]
                df["bb_pct"] = bbands.iloc[:, 4]  # %B

            # ATR
            df["atr"] = ta.atr(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["atr_period"],
            )

            # Keltner Channel
            kc = ta.kc(
                df["high"],
                df["low"],
                df["close"],
                length=self.params["kc_period"],
                scalar=self.params["kc_multiplier"],
            )
            if kc is not None and not kc.empty:
                df["kc_lower"] = kc.iloc[:, 0]
                df["kc_mid"] = kc.iloc[:, 1]
                df["kc_upper"] = kc.iloc[:, 2]

            # Donchian Channel
            donchian = ta.donchian(
                df["high"],
                df["low"],
                lower_length=self.params["donchian_period"],
                upper_length=self.params["donchian_period"],
            )
            if donchian is not None and not donchian.empty:
                df["dc_lower"] = donchian.iloc[:, 0]
                df["dc_mid"] = donchian.iloc[:, 1]
                df["dc_upper"] = donchian.iloc[:, 2]

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
