"""Strategy Engine Models.

Data models for the self-evolving agentic trading system:
- TradeRecord: Complete trade lifecycle (Episodic Memory)
- StrategyInsight: Extracted patterns (Semantic Memory)
- ParameterState: Strategy parameters (Procedural Memory)

Statistical Edge Collection System:
- Edge: Registered statistical edge with measurable properties
- EdgeSignal: Signal when edge conditions are met
- EdgeHealth: Health status of an edge
- PositionSize: Kelly-based position sizing result
"""

from .memory import (
    # Enums
    TradeStatus,
    ExitReason,
    PositionAction,
    # Core models
    TradeRecord,
    StrategyInsight,
    ParameterState,
    ParameterEvolution,
    # Agent outputs
    PositionReview,
    TradeScore,
    DailyLearningReport,
)

from .edge import (
    # Edge Enums
    EdgeStatus,
    EdgeType,
    EdgeHealthStatus,
    # Edge Core Models
    Edge,
    EdgeSignal,
    PositionSize,
    EdgeHealth,
    EdgeTradeAttribution,
    ConvergenceReport,
)

__all__ = [
    # Memory Enums
    "TradeStatus",
    "ExitReason",
    "PositionAction",
    # Memory Core models
    "TradeRecord",
    "StrategyInsight",
    "ParameterState",
    "ParameterEvolution",
    # Memory Agent outputs
    "PositionReview",
    "TradeScore",
    "DailyLearningReport",
    # Edge Enums
    "EdgeStatus",
    "EdgeType",
    "EdgeHealthStatus",
    # Edge Core Models
    "Edge",
    "EdgeSignal",
    "PositionSize",
    "EdgeHealth",
    "EdgeTradeAttribution",
    "ConvergenceReport",
]
