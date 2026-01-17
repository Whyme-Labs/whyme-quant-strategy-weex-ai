"""Strategy Engine Configuration.

Configuration modules for the edge-based trading system:
- symbols: Trading symbol configurations
- edges: Predefined edge definitions
"""

from .symbols import (
    SymbolConfig,
    TRADING_SYMBOLS,
    CORRELATION_LIMITS,
    get_symbol_config,
    get_enabled_symbols,
)
from .edges import (
    PREDEFINED_EDGES,
    get_edges_for_symbol,
    get_edges_by_type,
)

__all__ = [
    # Symbols
    "SymbolConfig",
    "TRADING_SYMBOLS",
    "CORRELATION_LIMITS",
    "get_symbol_config",
    "get_enabled_symbols",
    # Edges
    "PREDEFINED_EDGES",
    "get_edges_for_symbol",
    "get_edges_by_type",
]
