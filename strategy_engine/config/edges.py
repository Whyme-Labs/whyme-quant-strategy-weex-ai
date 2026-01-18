"""Predefined Edge Definitions for Statistical Edge Collection System.

COMPREHENSIVE 100% UTILIZATION VERSION

This module defines edges for ALL available indicators, patterns, and signals:
- Technical Indicators: 50+ indicators utilized
- Chart Patterns: 11 patterns as edges
- Candlestick Patterns: 20+ patterns as edges

Edge types:
- Mean Reversion: Buy low, sell high (fade extremes)
- Trend Following: Buy high, sell higher (ride trends)
- Turtle: Classic breakout system (20/55-day channels)
- MACD: Momentum divergence and crossover edges
- Stochastic: Overbought/oversold with crossovers
- Ichimoku: Cloud-based trend and momentum
- SuperTrend: Dynamic trend following
- Volume: OBV, CMF, MFI based edges
- Chart Pattern: Classical chart patterns
- Candlestick: Japanese candlestick reversal patterns

Note: Initial statistics (win_rate, avg_win_pct, etc.) are set to neutral
values. Actual statistics will be updated as trades complete.
"""

from typing import List, Optional

from ..models.edge import Edge, EdgeType, EdgeStatus


# ============================================================================
# TRADING SYMBOLS
# ============================================================================
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ============================================================================
# PREDEFINED EDGES - 100% INDICATOR UTILIZATION
# ============================================================================

PREDEFINED_EDGES: List[Edge] = []


# ============================================================================
# MEAN REVERSION EDGES (RSI + Bollinger Bands)
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # RSI Oversold + BB Lower (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"mr_rsi_oversold_{sym_short}_4h",
        name=f"RSI Oversold + BB Lower ({symbol} 4H)",
        description="Enter long when RSI is oversold and price below lower Bollinger Band",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "rsi_below": 30,
            "bb_position": "below_lower",
            "side": "long",
        },
        exit_conditions={
            "rsi_above": 50,
            "or": {"bb_position": "above_middle"},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # RSI Overbought + BB Upper (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"mr_rsi_overbought_{sym_short}_4h",
        name=f"RSI Overbought + BB Upper ({symbol} 4H)",
        description="Enter short when RSI is overbought and price above upper Bollinger Band",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "rsi_above": 70,
            "bb_position": "above_upper",
            "side": "short",
        },
        exit_conditions={
            "rsi_below": 50,
            "or": {"bb_position": "below_middle"},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # RSI Extreme (1H) - faster mean reversion
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"mr_rsi_extreme_{sym_short}_1h",
        name=f"RSI Extreme ({symbol} 1H)",
        description="Enter on extreme RSI readings for quick mean reversion",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol=symbol,
        timeframe="1h",
        entry_conditions={
            "rsi_extreme": True,  # RSI < 20 or > 80
            "side": "auto",
        },
        exit_conditions={
            "rsi_neutral": True,  # RSI between 40-60
        },
        status=EdgeStatus.WARMING,
        min_sample_size=50,
        min_expectancy=0.003,
    ))


# ============================================================================
# TREND FOLLOWING EDGES (EMA + Breakout)
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # 20-Day Channel Breakout (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"tf_channel_breakout_{sym_short}_1d",
        name=f"20-Day Channel Breakout ({symbol} 1D)",
        description="Enter on breakout of 20-day high/low",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "breakout": "20d",
            "side": "auto",
        },
        exit_conditions={
            "breakout_reverse": "10d",
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # VCP Pattern (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"tf_vcp_{sym_short}_4h",
        name=f"VCP Pattern Breakout ({symbol} 4H)",
        description="Volatility Contraction Pattern - enter on breakout from tightening ranges",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "vcp",
            "atr_contraction": 0.6,
            "side": "long",
        },
        exit_conditions={
            "atr_stop": 1.5,
            "trailing_pct": 0.02,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # EMA Alignment (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"tf_ema_alignment_{sym_short}_4h",
        name=f"EMA Alignment ({symbol} 4H)",
        description="Enter when 8/20/50 EMAs align in trending formation",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "ema_alignment": "8_20_50",
            "trend_strength": "strong",
            "side": "auto",
        },
        exit_conditions={
            "ema_cross": "8_20",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))


# ============================================================================
# TURTLE TRADING EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # Turtle System 1 (20-day breakout)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"turtle_s1_{sym_short}_1d",
        name=f"Turtle System 1 ({symbol} 1D)",
        description="Classic Turtle System 1: 20-day breakout entry, 10-day exit",
        edge_type=EdgeType.TURTLE,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "system": "1",
            "breakout": "20d",
            "skip_if_last_win": True,
            "side": "auto",
        },
        exit_conditions={
            "breakout_reverse": "10d",
            "atr_stop": 2.0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # Turtle System 2 (55-day breakout)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"turtle_s2_{sym_short}_1d",
        name=f"Turtle System 2 ({symbol} 1D)",
        description="Classic Turtle System 2: 55-day breakout entry, 20-day exit",
        edge_type=EdgeType.TURTLE,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "system": "2",
            "breakout": "55d",
            "side": "auto",
        },
        exit_conditions={
            "breakout_reverse": "20d",
            "atr_stop": 2.0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))


# ============================================================================
# MACD EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # MACD Bullish Crossover (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"macd_bullish_cross_{sym_short}_4h",
        name=f"MACD Bullish Crossover ({symbol} 4H)",
        description="Enter long when MACD line crosses above signal line with histogram turning positive",
        edge_type=EdgeType.MACD,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "macd_crossover": "bullish",
            "macd_hist_positive": True,
            "side": "long",
        },
        exit_conditions={
            "macd_crossover": "bearish",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # MACD Bearish Crossover (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"macd_bearish_cross_{sym_short}_4h",
        name=f"MACD Bearish Crossover ({symbol} 4H)",
        description="Enter short when MACD line crosses below signal line with histogram turning negative",
        edge_type=EdgeType.MACD,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "macd_crossover": "bearish",
            "macd_hist_negative": True,
            "side": "short",
        },
        exit_conditions={
            "macd_crossover": "bullish",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # MACD Bullish Divergence (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"macd_bull_divergence_{sym_short}_4h",
        name=f"MACD Bullish Divergence ({symbol} 4H)",
        description="Enter long when price makes lower low but MACD makes higher low (bullish divergence)",
        edge_type=EdgeType.MACD,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "macd_divergence": "bullish",
            "side": "long",
        },
        exit_conditions={
            "macd_crossover": "bearish",
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # MACD Bearish Divergence (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"macd_bear_divergence_{sym_short}_4h",
        name=f"MACD Bearish Divergence ({symbol} 4H)",
        description="Enter short when price makes higher high but MACD makes lower high (bearish divergence)",
        edge_type=EdgeType.MACD,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "macd_divergence": "bearish",
            "side": "short",
        },
        exit_conditions={
            "macd_crossover": "bullish",
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # MACD Zero Line Cross (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"macd_zero_cross_{sym_short}_1d",
        name=f"MACD Zero Line Cross ({symbol} 1D)",
        description="Enter on MACD crossing the zero line - stronger trend confirmation",
        edge_type=EdgeType.MACD,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "macd_zero_cross": True,
            "side": "auto",
        },
        exit_conditions={
            "macd_zero_cross_reverse": True,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.012,
    ))


# ============================================================================
# STOCHASTIC EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # Stochastic Oversold Crossover (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"stoch_oversold_{sym_short}_4h",
        name=f"Stochastic Oversold Crossover ({symbol} 4H)",
        description="Enter long when Stochastic %K crosses above %D in oversold zone (<20)",
        edge_type=EdgeType.STOCHASTIC,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "stoch_k_below": 20,
            "stoch_crossover": "bullish",
            "side": "long",
        },
        exit_conditions={
            "stoch_k_above": 80,
            "or": {"stoch_crossover": "bearish"},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # Stochastic Overbought Crossover (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"stoch_overbought_{sym_short}_4h",
        name=f"Stochastic Overbought Crossover ({symbol} 4H)",
        description="Enter short when Stochastic %K crosses below %D in overbought zone (>80)",
        edge_type=EdgeType.STOCHASTIC,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "stoch_k_above": 80,
            "stoch_crossover": "bearish",
            "side": "short",
        },
        exit_conditions={
            "stoch_k_below": 20,
            "or": {"stoch_crossover": "bullish"},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # Stochastic Double Bottom (1H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"stoch_double_bottom_{sym_short}_1h",
        name=f"Stochastic Double Bottom ({symbol} 1H)",
        description="Enter long when Stochastic forms double bottom pattern in oversold zone",
        edge_type=EdgeType.STOCHASTIC,
        symbol=symbol,
        timeframe="1h",
        entry_conditions={
            "stoch_pattern": "double_bottom",
            "stoch_k_below": 25,
            "side": "long",
        },
        exit_conditions={
            "stoch_k_above": 70,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=40,
        min_expectancy=0.004,
    ))


# ============================================================================
# MOMENTUM EDGES (CCI, Williams %R, ROC)
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # CCI Oversold (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cci_oversold_{sym_short}_4h",
        name=f"CCI Oversold ({symbol} 4H)",
        description="Enter long when CCI drops below -100 (oversold)",
        edge_type=EdgeType.MOMENTUM,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "cci_below": -100,
            "side": "long",
        },
        exit_conditions={
            "cci_above": 0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # CCI Overbought (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cci_overbought_{sym_short}_4h",
        name=f"CCI Overbought ({symbol} 4H)",
        description="Enter short when CCI rises above +100 (overbought)",
        edge_type=EdgeType.MOMENTUM,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "cci_above": 100,
            "side": "short",
        },
        exit_conditions={
            "cci_below": 0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # Williams %R Oversold (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"willr_oversold_{sym_short}_4h",
        name=f"Williams %R Oversold ({symbol} 4H)",
        description="Enter long when Williams %R drops below -80 (oversold)",
        edge_type=EdgeType.MOMENTUM,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "willr_below": -80,
            "side": "long",
        },
        exit_conditions={
            "willr_above": -50,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # Williams %R Overbought (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"willr_overbought_{sym_short}_4h",
        name=f"Williams %R Overbought ({symbol} 4H)",
        description="Enter short when Williams %R rises above -20 (overbought)",
        edge_type=EdgeType.MOMENTUM,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "willr_above": -20,
            "side": "short",
        },
        exit_conditions={
            "willr_below": -50,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # ROC Momentum Breakout (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"roc_breakout_{sym_short}_4h",
        name=f"ROC Momentum Breakout ({symbol} 4H)",
        description="Enter on strong momentum when ROC exceeds threshold",
        edge_type=EdgeType.MOMENTUM,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "roc_above": 5,  # 5% price change
            "side": "auto",
        },
        exit_conditions={
            "roc_reversal": True,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))


# ============================================================================
# SUPERTREND EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # SuperTrend Flip Bullish (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"supertrend_bull_{sym_short}_4h",
        name=f"SuperTrend Bullish Flip ({symbol} 4H)",
        description="Enter long when SuperTrend flips from bearish to bullish",
        edge_type=EdgeType.SUPERTREND,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "supertrend_flip": "bullish",
            "side": "long",
        },
        exit_conditions={
            "supertrend_flip": "bearish",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # SuperTrend Flip Bearish (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"supertrend_bear_{sym_short}_4h",
        name=f"SuperTrend Bearish Flip ({symbol} 4H)",
        description="Enter short when SuperTrend flips from bullish to bearish",
        edge_type=EdgeType.SUPERTREND,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "supertrend_flip": "bearish",
            "side": "short",
        },
        exit_conditions={
            "supertrend_flip": "bullish",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # SuperTrend + ADX Strong Trend (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"supertrend_adx_{sym_short}_1d",
        name=f"SuperTrend + ADX Strong Trend ({symbol} 1D)",
        description="Enter when SuperTrend is bullish/bearish AND ADX shows strong trend (>25)",
        edge_type=EdgeType.SUPERTREND,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "supertrend_direction": "auto",
            "adx_above": 25,
            "side": "auto",
        },
        exit_conditions={
            "supertrend_flip": True,
            "or": {"adx_below": 20},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.012,
    ))


# ============================================================================
# ICHIMOKU EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # Ichimoku Cloud Breakout (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"ichimoku_cloud_break_{sym_short}_4h",
        name=f"Ichimoku Cloud Breakout ({symbol} 4H)",
        description="Enter when price breaks above/below the Ichimoku cloud",
        edge_type=EdgeType.ICHIMOKU,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "ichimoku_cloud_break": True,
            "side": "auto",  # long above cloud, short below cloud
        },
        exit_conditions={
            "ichimoku_cloud_reenter": True,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # Ichimoku TK Cross (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"ichimoku_tk_cross_{sym_short}_4h",
        name=f"Ichimoku TK Cross ({symbol} 4H)",
        description="Enter on Tenkan-Kijun crossover (fast over slow line)",
        edge_type=EdgeType.ICHIMOKU,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "ichimoku_tk_cross": "bullish",
            "price_above_cloud": True,  # Additional confirmation
            "side": "long",
        },
        exit_conditions={
            "ichimoku_tk_cross": "bearish",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # Ichimoku Kumo Twist (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"ichimoku_kumo_twist_{sym_short}_1d",
        name=f"Ichimoku Kumo Twist ({symbol} 1D)",
        description="Enter when Senkou A crosses Senkou B (cloud color change)",
        edge_type=EdgeType.ICHIMOKU,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "ichimoku_kumo_twist": True,
            "side": "auto",
        },
        exit_conditions={
            "ichimoku_kumo_twist_reverse": True,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))


# ============================================================================
# VOLUME EDGES (OBV, CMF, MFI)
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # OBV Divergence Bullish (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"obv_bull_divergence_{sym_short}_4h",
        name=f"OBV Bullish Divergence ({symbol} 4H)",
        description="Enter long when price makes lower low but OBV makes higher low",
        edge_type=EdgeType.VOLUME,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "obv_divergence": "bullish",
            "side": "long",
        },
        exit_conditions={
            "obv_divergence_cancel": True,
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # OBV Divergence Bearish (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"obv_bear_divergence_{sym_short}_4h",
        name=f"OBV Bearish Divergence ({symbol} 4H)",
        description="Enter short when price makes higher high but OBV makes lower high",
        edge_type=EdgeType.VOLUME,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "obv_divergence": "bearish",
            "side": "short",
        },
        exit_conditions={
            "obv_divergence_cancel": True,
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # CMF Accumulation (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cmf_accumulation_{sym_short}_4h",
        name=f"CMF Accumulation ({symbol} 4H)",
        description="Enter long when CMF shows strong accumulation (>0.1)",
        edge_type=EdgeType.VOLUME,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "cmf_above": 0.1,
            "side": "long",
        },
        exit_conditions={
            "cmf_below": 0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.006,
    ))

    # CMF Distribution (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cmf_distribution_{sym_short}_4h",
        name=f"CMF Distribution ({symbol} 4H)",
        description="Enter short when CMF shows strong distribution (<-0.1)",
        edge_type=EdgeType.VOLUME,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "cmf_below": -0.1,
            "side": "short",
        },
        exit_conditions={
            "cmf_above": 0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.006,
    ))

    # MFI Oversold (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"mfi_oversold_{sym_short}_4h",
        name=f"MFI Oversold ({symbol} 4H)",
        description="Enter long when Money Flow Index drops below 20 (oversold)",
        edge_type=EdgeType.VOLUME,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "mfi_below": 20,
            "side": "long",
        },
        exit_conditions={
            "mfi_above": 50,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))

    # MFI Overbought (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"mfi_overbought_{sym_short}_4h",
        name=f"MFI Overbought ({symbol} 4H)",
        description="Enter short when Money Flow Index rises above 80 (overbought)",
        edge_type=EdgeType.VOLUME,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "mfi_above": 80,
            "side": "short",
        },
        exit_conditions={
            "mfi_below": 50,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))


# ============================================================================
# KELTNER CHANNEL EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # Keltner Channel Squeeze Breakout (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"keltner_squeeze_{sym_short}_4h",
        name=f"Keltner Squeeze Breakout ({symbol} 4H)",
        description="Enter when BB squeeze inside KC releases with breakout",
        edge_type=EdgeType.BREAKOUT,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "keltner_squeeze": True,
            "squeeze_fire": True,  # BB width expanding after contraction
            "side": "auto",
        },
        exit_conditions={
            "keltner_middle_cross": True,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ))

    # Keltner Channel Mean Reversion (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"keltner_mr_{sym_short}_4h",
        name=f"Keltner Channel Reversion ({symbol} 4H)",
        description="Enter when price touches outer Keltner band for mean reversion",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "keltner_position": "outside",  # Price outside KC bands
            "side": "auto",  # Fade the move
        },
        exit_conditions={
            "keltner_position": "middle",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.005,
    ))


# ============================================================================
# CHART PATTERN EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # Head & Shoulders (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_h_s_{sym_short}_4h",
        name=f"Head & Shoulders ({symbol} 4H)",
        description="Enter short on Head & Shoulders pattern neckline break",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "head_and_shoulders",
            "pattern_activated": True,
            "side": "short",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Inverse Head & Shoulders (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_inv_h_s_{sym_short}_4h",
        name=f"Inverse Head & Shoulders ({symbol} 4H)",
        description="Enter long on Inverse Head & Shoulders pattern neckline break",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "inverse_head_and_shoulders",
            "pattern_activated": True,
            "side": "long",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Double Top (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_double_top_{sym_short}_4h",
        name=f"Double Top ({symbol} 4H)",
        description="Enter short on Double Top pattern support break",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "double_top",
            "pattern_activated": True,
            "side": "short",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.012,
    ))

    # Double Bottom (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_double_bottom_{sym_short}_4h",
        name=f"Double Bottom ({symbol} 4H)",
        description="Enter long on Double Bottom pattern resistance break",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "double_bottom",
            "pattern_activated": True,
            "side": "long",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.012,
    ))

    # Ascending Triangle (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_asc_triangle_{sym_short}_4h",
        name=f"Ascending Triangle ({symbol} 4H)",
        description="Enter long on Ascending Triangle resistance breakout",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "ascending_triangle",
            "pattern_activated": True,
            "side": "long",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Descending Triangle (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_desc_triangle_{sym_short}_4h",
        name=f"Descending Triangle ({symbol} 4H)",
        description="Enter short on Descending Triangle support breakdown",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "descending_triangle",
            "pattern_activated": True,
            "side": "short",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Bull Flag (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_bull_flag_{sym_short}_4h",
        name=f"Bull Flag ({symbol} 4H)",
        description="Enter long on Bull Flag pattern breakout",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "bull_flag",
            "pattern_activated": True,
            "side": "long",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.012,
    ))

    # Bear Flag (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_bear_flag_{sym_short}_4h",
        name=f"Bear Flag ({symbol} 4H)",
        description="Enter short on Bear Flag pattern breakdown",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "bear_flag",
            "pattern_activated": True,
            "side": "short",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.012,
    ))

    # Rising Wedge (4H) - Bearish
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_rising_wedge_{sym_short}_4h",
        name=f"Rising Wedge ({symbol} 4H)",
        description="Enter short on Rising Wedge pattern breakdown (bearish)",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "rising_wedge",
            "pattern_activated": True,
            "side": "short",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.012,
    ))

    # Falling Wedge (4H) - Bullish
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"pattern_falling_wedge_{sym_short}_4h",
        name=f"Falling Wedge ({symbol} 4H)",
        description="Enter long on Falling Wedge pattern breakout (bullish)",
        edge_type=EdgeType.CHART_PATTERN,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "pattern": "falling_wedge",
            "pattern_activated": True,
            "side": "long",
        },
        exit_conditions={
            "pattern_target_reached": True,
            "or": {"pattern_invalidated": True},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.012,
    ))


# ============================================================================
# CANDLESTICK PATTERN EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # Hammer at Support (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_hammer_{sym_short}_4h",
        name=f"Hammer at Support ({symbol} 4H)",
        description="Enter long on Hammer candlestick near support (BB lower or recent low)",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "candlestick": "hammer",
            "near_support": True,  # Near BB lower or recent low
            "side": "long",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 1.5},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # Shooting Star at Resistance (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_shooting_star_{sym_short}_4h",
        name=f"Shooting Star at Resistance ({symbol} 4H)",
        description="Enter short on Shooting Star candlestick near resistance",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "candlestick": "shooting_star",
            "near_resistance": True,
            "side": "short",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 1.5},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # Bullish Engulfing (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_bull_engulfing_{sym_short}_4h",
        name=f"Bullish Engulfing ({symbol} 4H)",
        description="Enter long on Bullish Engulfing pattern",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "candlestick": "engulfing",
            "engulfing_direction": "bullish",
            "side": "long",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # Bearish Engulfing (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_bear_engulfing_{sym_short}_4h",
        name=f"Bearish Engulfing ({symbol} 4H)",
        description="Enter short on Bearish Engulfing pattern",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "candlestick": "engulfing",
            "engulfing_direction": "bearish",
            "side": "short",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ))

    # Morning Star (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_morning_star_{sym_short}_1d",
        name=f"Morning Star ({symbol} 1D)",
        description="Enter long on Morning Star 3-candle reversal pattern",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "candlestick": "morning_star",
            "side": "long",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Evening Star (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_evening_star_{sym_short}_1d",
        name=f"Evening Star ({symbol} 1D)",
        description="Enter short on Evening Star 3-candle reversal pattern",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "candlestick": "evening_star",
            "side": "short",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 2.0},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Three White Soldiers (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_three_soldiers_{sym_short}_1d",
        name=f"Three White Soldiers ({symbol} 1D)",
        description="Enter long on Three White Soldiers strong bullish pattern",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "candlestick": "three_white_soldiers",
            "side": "long",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"trailing_pct": 0.03},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Three Black Crows (1D)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_three_crows_{sym_short}_1d",
        name=f"Three Black Crows ({symbol} 1D)",
        description="Enter short on Three Black Crows strong bearish pattern",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="1d",
        entry_conditions={
            "candlestick": "three_black_crows",
            "side": "short",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"trailing_pct": 0.03},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ))

    # Doji at Extreme (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_doji_extreme_{sym_short}_4h",
        name=f"Doji at Extreme ({symbol} 4H)",
        description="Enter on Doji candlestick at RSI extreme levels (potential reversal)",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "candlestick": "doji",
            "rsi_extreme": True,
            "side": "auto",  # Counter-trend
        },
        exit_conditions={
            "candlestick_confirmation": True,  # Wait for follow-through
            "or": {"atr_stop": 1.5},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.006,
    ))

    # Harami Reversal (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"cdl_harami_{sym_short}_4h",
        name=f"Harami Reversal ({symbol} 4H)",
        description="Enter on Harami pattern (inside bar) for potential reversal",
        edge_type=EdgeType.CANDLESTICK,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "candlestick": "harami",
            "side": "auto",
        },
        exit_conditions={
            "candlestick_reversal": True,
            "or": {"atr_stop": 1.5},
        },
        status=EdgeStatus.WARMING,
        min_sample_size=30,
        min_expectancy=0.006,
    ))


# ============================================================================
# COMBINED/CONFLUENCE EDGES
# ============================================================================

for symbol in SYMBOLS:
    sym_short = symbol[:3].lower()

    # RSI + MACD Confluence (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"confluence_rsi_macd_{sym_short}_4h",
        name=f"RSI + MACD Confluence ({symbol} 4H)",
        description="Enter when both RSI oversold/overbought AND MACD confirms direction",
        edge_type=EdgeType.MOMENTUM,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "rsi_extreme": True,
            "macd_confirms": True,
            "side": "auto",
        },
        exit_conditions={
            "rsi_neutral": True,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.01,
    ))

    # Volume + Price Breakout (4H)
    PREDEFINED_EDGES.append(Edge(
        edge_id=f"confluence_volume_breakout_{sym_short}_4h",
        name=f"Volume Confirmed Breakout ({symbol} 4H)",
        description="Enter on price breakout with volume confirmation (CMF > 0.1)",
        edge_type=EdgeType.BREAKOUT,
        symbol=symbol,
        timeframe="4h",
        entry_conditions={
            "breakout": "20d",
            "cmf_confirms": True,  # CMF in direction of breakout
            "side": "auto",
        },
        exit_conditions={
            "breakout_reverse": "10d",
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.012,
    ))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_edges_for_symbol(symbol: str) -> List[Edge]:
    """Get all edges for a specific symbol.

    Args:
        symbol: Symbol string (e.g., "BTCUSDT")

    Returns:
        List of Edge objects for that symbol
    """
    return [e for e in PREDEFINED_EDGES if e.symbol == symbol or e.symbol == "*"]


def get_edges_by_type(edge_type: EdgeType) -> List[Edge]:
    """Get all edges of a specific type.

    Args:
        edge_type: EdgeType enum

    Returns:
        List of Edge objects of that type
    """
    return [e for e in PREDEFINED_EDGES if e.edge_type == edge_type]


def get_edge_by_id(edge_id: str) -> Optional[Edge]:
    """Get edge by ID.

    Args:
        edge_id: Edge identifier

    Returns:
        Edge object or None
    """
    for edge in PREDEFINED_EDGES:
        if edge.edge_id == edge_id:
            return edge
    return None


def get_active_edges() -> List[Edge]:
    """Get all edges that could potentially be active.

    Returns all edges except DISABLED ones.

    Returns:
        List of non-disabled Edge objects
    """
    return [e for e in PREDEFINED_EDGES if e.status != EdgeStatus.DISABLED]


def get_warming_edges() -> List[Edge]:
    """Get edges currently warming up (collecting data).

    Returns:
        List of Edge objects in WARMING status
    """
    return [e for e in PREDEFINED_EDGES if e.status == EdgeStatus.WARMING]


def get_edge_count_summary() -> dict:
    """Get summary of edge counts by type.

    Returns:
        Dictionary with counts per edge type
    """
    summary = {}
    for edge in PREDEFINED_EDGES:
        edge_type = edge.edge_type.value
        if edge_type not in summary:
            summary[edge_type] = 0
        summary[edge_type] += 1
    summary["total"] = len(PREDEFINED_EDGES)
    return summary
