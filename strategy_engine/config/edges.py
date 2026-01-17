"""Predefined Edge Definitions for Statistical Edge Collection System.

Registers existing strategies as edges with measurable statistical properties.
Each edge represents a specific pattern/condition that has historically
provided a statistical advantage.

Edge types:
- Mean Reversion: Buy low, sell high (fade extremes)
- Trend Following: Buy high, sell higher (ride trends)
- Turtle: Classic breakout system (20/55-day channels)

Note: Initial statistics (win_rate, avg_win_pct, etc.) are set to neutral
values. Actual statistics will be updated as trades complete.
"""

from typing import List, Optional

from ..models.edge import Edge, EdgeType, EdgeStatus


# ============================================================================
# PREDEFINED EDGES
# ============================================================================

PREDEFINED_EDGES: List[Edge] = [
    # ========================================================================
    # MEAN REVERSION EDGES - BTC
    # ========================================================================
    Edge(
        edge_id="mr_rsi_oversold_btc_4h",
        name="RSI Oversold + BB Lower (BTC 4H)",
        description="Enter long when RSI is oversold and price below lower Bollinger Band on 4H timeframe",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol="BTCUSDT",
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
        min_expectancy=0.005,  # 0.5% minimum edge
    ),
    Edge(
        edge_id="mr_rsi_overbought_btc_4h",
        name="RSI Overbought + BB Upper (BTC 4H)",
        description="Enter short when RSI is overbought and price above upper Bollinger Band on 4H timeframe",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol="BTCUSDT",
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
    ),
    Edge(
        edge_id="mr_rsi_extreme_btc_1h",
        name="RSI Extreme (BTC 1H)",
        description="Enter on extreme RSI readings on 1H timeframe for quick mean reversion",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol="BTCUSDT",
        timeframe="1h",
        entry_conditions={
            "rsi_extreme": True,  # RSI < 20 or > 80
            "side": "auto",  # Long if oversold, short if overbought
        },
        exit_conditions={
            "rsi_neutral": True,  # RSI between 40-60
        },
        status=EdgeStatus.WARMING,
        min_sample_size=50,  # Higher sample for 1H (more trades expected)
        min_expectancy=0.003,  # Lower edge acceptable for higher frequency
    ),

    # ========================================================================
    # MEAN REVERSION EDGES - ETH
    # ========================================================================
    Edge(
        edge_id="mr_rsi_oversold_eth_4h",
        name="RSI Oversold + BB Lower (ETH 4H)",
        description="Enter long when RSI is oversold and price below lower Bollinger Band on 4H timeframe",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol="ETHUSDT",
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
    ),
    Edge(
        edge_id="mr_rsi_overbought_eth_4h",
        name="RSI Overbought + BB Upper (ETH 4H)",
        description="Enter short when RSI is overbought and price above upper Bollinger Band on 4H timeframe",
        edge_type=EdgeType.MEAN_REVERSION,
        symbol="ETHUSDT",
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
    ),

    # ========================================================================
    # TREND FOLLOWING EDGES - BTC
    # ========================================================================
    Edge(
        edge_id="tf_channel_breakout_btc_1d",
        name="20-Day Channel Breakout (BTC 1D)",
        description="Enter long on breakout above 20-day high, short on breakdown below 20-day low",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol="BTCUSDT",
        timeframe="1d",
        entry_conditions={
            "breakout": "20d",  # 20-day high/low breakout
            "side": "auto",     # Long on high breakout, short on low breakdown
        },
        exit_conditions={
            "breakout_reverse": "10d",  # Exit on 10-day reversal
            "or": {"atr_stop": 2.0},     # 2 ATR stop loss
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,  # Fewer trades on daily
        min_expectancy=0.01,  # Higher edge expected for trend following
    ),
    Edge(
        edge_id="tf_vcp_btc_4h",
        name="VCP Pattern Breakout (BTC 4H)",
        description="Volatility Contraction Pattern - enter on breakout from tightening ranges",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol="BTCUSDT",
        timeframe="4h",
        entry_conditions={
            "pattern": "vcp",
            "atr_contraction": 0.6,  # ATR contracted to 60% of recent avg
            "side": "long",
        },
        exit_conditions={
            "atr_stop": 1.5,  # 1.5 ATR stop
            "trailing_pct": 0.02,  # 2% trailing stop once in profit
        },
        status=EdgeStatus.WARMING,
        min_sample_size=15,
        min_expectancy=0.015,
    ),
    Edge(
        edge_id="tf_ema_alignment_btc_4h",
        name="EMA Alignment (BTC 4H)",
        description="Enter when 8/20/50 EMAs align in trending formation",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol="BTCUSDT",
        timeframe="4h",
        entry_conditions={
            "ema_alignment": "8_20_50",
            "trend_strength": "strong",  # All EMAs sloping in same direction
            "side": "auto",
        },
        exit_conditions={
            "ema_cross": "8_20",  # Exit on 8/20 EMA cross against
        },
        status=EdgeStatus.WARMING,
        min_sample_size=25,
        min_expectancy=0.008,
    ),

    # ========================================================================
    # TREND FOLLOWING EDGES - ETH
    # ========================================================================
    Edge(
        edge_id="tf_channel_breakout_eth_1d",
        name="20-Day Channel Breakout (ETH 1D)",
        description="Enter long on breakout above 20-day high, short on breakdown below 20-day low",
        edge_type=EdgeType.TREND_FOLLOWING,
        symbol="ETHUSDT",
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
    ),

    # ========================================================================
    # TURTLE TRADING EDGES - BTC
    # ========================================================================
    Edge(
        edge_id="turtle_s1_btc_1d",
        name="Turtle System 1 (BTC 1D)",
        description="Classic Turtle System 1: 20-day breakout entry, 10-day exit",
        edge_type=EdgeType.TURTLE,
        symbol="BTCUSDT",
        timeframe="1d",
        entry_conditions={
            "system": "1",
            "breakout": "20d",
            "skip_if_last_win": True,  # Skip if last S1 trade was winner
            "side": "auto",
        },
        exit_conditions={
            "breakout_reverse": "10d",
            "atr_stop": 2.0,
        },
        status=EdgeStatus.WARMING,
        min_sample_size=20,
        min_expectancy=0.01,
    ),
    Edge(
        edge_id="turtle_s2_btc_1d",
        name="Turtle System 2 (BTC 1D)",
        description="Classic Turtle System 2: 55-day breakout entry, 20-day exit",
        edge_type=EdgeType.TURTLE,
        symbol="BTCUSDT",
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
        min_sample_size=15,  # Fewer trades for 55-day system
        min_expectancy=0.015,  # Higher edge expected
    ),

    # ========================================================================
    # TURTLE TRADING EDGES - ETH
    # ========================================================================
    Edge(
        edge_id="turtle_s1_eth_1d",
        name="Turtle System 1 (ETH 1D)",
        description="Classic Turtle System 1: 20-day breakout entry, 10-day exit",
        edge_type=EdgeType.TURTLE,
        symbol="ETHUSDT",
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
    ),
]


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
