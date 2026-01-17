"""Trading Symbol Configuration for Multi-Symbol Support.

Defines the symbols available for trading with their configurations:
- Symbol mappings (internal to WEEX format)
- Position limits per symbol
- Correlation groups for diversification
- Contract specifications
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class SymbolConfig(BaseModel):
    """Configuration for a tradeable symbol."""
    symbol: str                       # Internal symbol (e.g., "BTCUSDT")
    weex_symbol: str                  # WEEX API symbol (e.g., "cmt_btcusdt")
    name: str                         # Human readable name
    enabled: bool = True              # Whether this symbol is enabled for trading

    # Position limits
    max_position_pct: float = 0.15    # Max position as % of portfolio
    max_leverage: int = 20            # Max leverage (hackathon limit)

    # Diversification
    correlation_group: str            # For correlation limits (e.g., "btc", "eth")

    # Contract specs
    tick_size: float = 0.01           # Minimum price increment
    lot_size: float = 0.001           # Minimum order size
    min_order_usd: float = 5.0        # Minimum order value in USD

    # Timeframes to analyze
    timeframes: List[str] = ["1h", "4h", "1d"]

    class Config:
        use_enum_values = True


# ============================================================================
# TRADING SYMBOLS
# ============================================================================

TRADING_SYMBOLS: List[SymbolConfig] = [
    SymbolConfig(
        symbol="BTCUSDT",
        weex_symbol="cmt_btcusdt",
        name="Bitcoin",
        enabled=True,
        max_position_pct=0.15,
        max_leverage=20,
        correlation_group="btc",
        tick_size=0.01,
        lot_size=0.001,
        min_order_usd=5.0,
        timeframes=["1h", "4h", "1d"],
    ),
    SymbolConfig(
        symbol="ETHUSDT",
        weex_symbol="cmt_ethusdt",
        name="Ethereum",
        enabled=True,
        max_position_pct=0.10,
        max_leverage=20,
        correlation_group="eth",
        tick_size=0.01,
        lot_size=0.01,
        min_order_usd=5.0,
        timeframes=["1h", "4h", "1d"],
    ),
    # Additional symbols can be added here
    SymbolConfig(
        symbol="SOLUSDT",
        weex_symbol="cmt_solusdt",
        name="Solana",
        enabled=False,  # Disabled by default - enable when ready
        max_position_pct=0.08,
        max_leverage=20,
        correlation_group="sol",
        tick_size=0.001,
        lot_size=0.1,
        min_order_usd=5.0,
        timeframes=["1h", "4h", "1d"],
    ),
    SymbolConfig(
        symbol="XRPUSDT",
        weex_symbol="cmt_xrpusdt",
        name="Ripple",
        enabled=False,  # Disabled by default
        max_position_pct=0.05,
        max_leverage=20,
        correlation_group="xrp",
        tick_size=0.0001,
        lot_size=1.0,
        min_order_usd=5.0,
        timeframes=["1h", "4h", "1d"],
    ),
]


# ============================================================================
# CORRELATION LIMITS
# ============================================================================

# Maximum exposure per correlation group and total
CORRELATION_LIMITS: Dict[str, float] = {
    "btc": 0.20,      # Max 20% in BTC-correlated assets
    "eth": 0.15,      # Max 15% in ETH-correlated assets
    "sol": 0.10,      # Max 10% in SOL-correlated assets
    "xrp": 0.08,      # Max 8% in XRP-correlated assets
    "total": 0.50,    # Max 50% total portfolio exposure
}


# Correlation matrix between assets (for correlation adjustment)
# Values are correlation coefficients (0-1)
ASSET_CORRELATIONS: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {
        "ETHUSDT": 0.85,
        "SOLUSDT": 0.75,
        "XRPUSDT": 0.70,
    },
    "ETHUSDT": {
        "BTCUSDT": 0.85,
        "SOLUSDT": 0.80,
        "XRPUSDT": 0.65,
    },
    "SOLUSDT": {
        "BTCUSDT": 0.75,
        "ETHUSDT": 0.80,
        "XRPUSDT": 0.60,
    },
    "XRPUSDT": {
        "BTCUSDT": 0.70,
        "ETHUSDT": 0.65,
        "SOLUSDT": 0.60,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_symbol_config(symbol: str) -> Optional[SymbolConfig]:
    """Get configuration for a symbol.

    Args:
        symbol: Symbol string (e.g., "BTCUSDT")

    Returns:
        SymbolConfig or None if not found
    """
    for config in TRADING_SYMBOLS:
        if config.symbol == symbol:
            return config
    return None


def get_enabled_symbols() -> List[SymbolConfig]:
    """Get all enabled trading symbols.

    Returns:
        List of enabled SymbolConfig objects
    """
    return [s for s in TRADING_SYMBOLS if s.enabled]


def get_weex_symbol(symbol: str) -> Optional[str]:
    """Get WEEX API symbol for internal symbol.

    Args:
        symbol: Internal symbol (e.g., "BTCUSDT")

    Returns:
        WEEX symbol (e.g., "cmt_btcusdt") or None
    """
    config = get_symbol_config(symbol)
    return config.weex_symbol if config else None


def get_correlation(symbol1: str, symbol2: str) -> float:
    """Get correlation between two symbols.

    Args:
        symbol1: First symbol
        symbol2: Second symbol

    Returns:
        Correlation coefficient (0-1), defaults to 0.5 if unknown
    """
    if symbol1 == symbol2:
        return 1.0

    if symbol1 in ASSET_CORRELATIONS:
        if symbol2 in ASSET_CORRELATIONS[symbol1]:
            return ASSET_CORRELATIONS[symbol1][symbol2]

    if symbol2 in ASSET_CORRELATIONS:
        if symbol1 in ASSET_CORRELATIONS[symbol2]:
            return ASSET_CORRELATIONS[symbol2][symbol1]

    # Default correlation for unknown pairs
    return 0.5


def get_correlation_limit(group: str) -> float:
    """Get correlation limit for a group.

    Args:
        group: Correlation group (e.g., "btc", "eth")

    Returns:
        Maximum exposure fraction for the group
    """
    return CORRELATION_LIMITS.get(group, 0.10)


def get_total_exposure_limit() -> float:
    """Get total portfolio exposure limit.

    Returns:
        Maximum total exposure fraction
    """
    return CORRELATION_LIMITS.get("total", 0.50)
