"""Strategy Engine Services.

Services provide shared functionality across agents:
- RedisClient: Async Redis client for cache persistence
- MarketDataService: Fetches and caches real OHLCV candle data
- IndicatorsService: Centralized technical indicator calculations
- PatternDetector: Chart pattern detection
- AlphaGenerator: Signal aggregation and alpha scoring
- TradeMemoryService: Triple memory system for self-evolving RL
- TradeJournal: Human-readable trade journaling like a professional trader

Statistical Edge Collection System:
- EdgeRegistry: Manages all registered edges and their statistics
- KellySizer: Position sizing based on Kelly Criterion
- EdgeScanner: Scans market for edge signals
- PerformanceTracker: Tracks expected vs actual performance
- EdgeHealthMonitor: Monitors edge health and auto-disable
"""

from .redis_client import RedisClient
from .market_data_service import MarketDataService
from .indicators_service import IndicatorsService
from .pattern_detector import PatternDetector
from .alpha_generator import AlphaGenerator
from .trade_memory import TradeMemoryService
from .trade_journal import TradeJournal
from .edge_registry import EdgeRegistry
from .kelly_sizer import KellySizer
from .edge_scanner import EdgeScanner
from .performance_tracker import PerformanceTracker
from .key_level_detector import KeyLevelDetector
from .smc_detector import SMCDetector

__all__ = [
    "RedisClient",
    "MarketDataService",
    "IndicatorsService",
    "PatternDetector",
    "AlphaGenerator",
    "TradeMemoryService",
    "TradeJournal",
    # Edge Collection System
    "EdgeRegistry",
    "KellySizer",
    "EdgeScanner",
    "PerformanceTracker",
    # Support/Resistance Detection
    "KeyLevelDetector",
    # Smart Money Concepts
    "SMCDetector",
]
