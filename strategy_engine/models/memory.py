"""Memory Models for Self-Evolving Agentic RL System.

This module defines the core data models for the triple memory system:
- Episodic Memory: TradeRecord (complete trade lifecycle)
- Semantic Memory: StrategyInsight (extracted patterns)
- Procedural Memory: ParameterState, ParameterEvolution (strategy parameters)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TradeStatus(str, Enum):
    """Status of a trade in its lifecycle."""
    PENDING = "pending"      # Signal generated, not yet executed
    OPEN = "open"           # Position is open
    CLOSED = "closed"       # Position closed
    CANCELLED = "cancelled"  # Order cancelled before fill


class ExitReason(str, Enum):
    """Reason for closing a position."""
    TAKE_PROFIT = "take_profit"      # Hit TP target
    STOP_LOSS = "stop_loss"          # Hit SL
    TRAILING_STOP = "trailing_stop"  # Trailing stop triggered
    REGIME_CHANGE = "regime_change"  # Market regime changed
    THESIS_BROKEN = "thesis_broken"  # Original thesis no longer valid
    MANUAL = "manual"                # Manual close
    REFLECTION = "reflection"        # Closed by reflection agent
    TIMEOUT = "timeout"              # Max hold time exceeded
    LIQUIDATION = "liquidation"      # Forced liquidation


class PositionAction(str, Enum):
    """Action suggested by reflection agent."""
    HOLD = "hold"           # Keep position as-is
    CLOSE = "close"         # Close entire position
    REDUCE = "reduce"       # Reduce position size
    ADD = "add"             # Pyramid/add to position
    ADJUST_STOP = "adjust_stop"  # Move stop loss


class TradeRecord(BaseModel):
    """Complete trade lifecycle record - stored in Episodic Memory.

    This captures everything about a trade from signal generation to exit,
    including the market context, AI decisions, and post-trade reflection.
    """
    trade_id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")

    # Entry information
    entry_timestamp: datetime = Field(..., description="Time of entry")
    entry_price: float = Field(..., description="Entry price")
    entry_side: str = Field(..., description="'long' or 'short'")
    entry_size: float = Field(..., description="Position size")
    entry_strategy: str = Field(..., description="Strategy that generated signal")
    entry_regime: Dict[str, str] = Field(
        default_factory=dict,
        description="Market regime at entry: {volatility, trend, volume}"
    )
    entry_confidence: float = Field(..., ge=0, le=1, description="Signal confidence 0-1")
    entry_reasoning: str = Field(..., description="Reasoning for entry")

    # Context at entry (for learning)
    market_data_at_entry: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market data snapshot at entry"
    )
    agent_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All AI agent decisions in the pipeline"
    )

    # Order tracking
    entry_order_id: Optional[str] = Field(None, description="Exchange order ID")
    exit_order_id: Optional[str] = Field(None, description="Exit order ID")

    # Exit information (filled when closed)
    exit_timestamp: Optional[datetime] = Field(None, description="Time of exit")
    exit_price: Optional[float] = Field(None, description="Exit price")
    exit_reason: Optional[ExitReason] = Field(None, description="Why position was closed")
    exit_regime: Optional[Dict[str, str]] = Field(None, description="Market regime at exit")

    # Outcome metrics
    pnl: Optional[float] = Field(None, description="Absolute P&L in USD")
    pnl_pct: Optional[float] = Field(None, description="Return percentage")
    duration_seconds: Optional[int] = Field(None, description="Trade duration")
    max_favorable_excursion: Optional[float] = Field(
        None, description="Best unrealized P&L during trade (MFE)"
    )
    max_adverse_excursion: Optional[float] = Field(
        None, description="Worst unrealized P&L during trade (MAE)"
    )

    # Reflection (filled by LLM after close)
    reflection: Optional[str] = Field(None, description="LLM reflection on trade")
    lessons_learned: Optional[List[str]] = Field(
        None, description="Actionable lessons extracted"
    )
    score: Optional[float] = Field(
        None, ge=0, le=100, description="Judge agent score 0-100"
    )
    score_breakdown: Optional[Dict[str, float]] = Field(
        None, description="Score by category"
    )

    status: TradeStatus = Field(default=TradeStatus.PENDING, description="Trade status")

    class Config:
        use_enum_values = True


class StrategyInsight(BaseModel):
    """Extracted pattern from trade analysis - stored in Semantic Memory.

    Insights represent learned knowledge about market behavior and strategy
    performance under specific conditions.
    """
    insight_id: str = Field(..., description="Unique insight identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # What was observed
    pattern: str = Field(
        ...,
        description="Pattern identifier (e.g., 'mean_reversion_fails_high_vol')"
    )
    description: str = Field(..., description="Natural language description")

    # Evidence
    supporting_trades: List[str] = Field(
        default_factory=list,
        description="Trade IDs that support this insight"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence in insight 0-1")
    sample_size: int = Field(default=0, description="Number of supporting observations")

    # When this insight applies
    applies_when: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for applying insight: {regime: {}, strategy: ''}"
    )

    # Recommendation
    recommendation: str = Field(..., description="What to do differently")
    affected_parameters: List[str] = Field(
        default_factory=list,
        description="Which parameters should be adjusted"
    )

    # Tracking
    times_applied: int = Field(default=0, description="How often this insight was used")
    success_rate: Optional[float] = Field(
        None, description="Success rate when following this insight"
    )


class ParameterState(BaseModel):
    """Current state of a strategy parameter - stored in Procedural Memory.

    Tracks the current value, bounds, and history of each tunable parameter.
    """
    agent: str = Field(..., description="Agent name (e.g., 'portfolio_manager')")
    parameter: str = Field(..., description="Parameter name (e.g., 'confidence_threshold')")

    current_value: Any = Field(..., description="Current parameter value")
    default_value: Any = Field(..., description="Original default value")

    # Bounds for autonomous evolution
    min_bound: Optional[Any] = Field(None, description="Minimum allowed value")
    max_bound: Optional[Any] = Field(None, description="Maximum allowed value")

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    update_reason: str = Field(default="initial", description="Reason for last update")

    # History of changes
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Change history: [{value, timestamp, reason, performance_after}]"
    )


class ParameterEvolution(BaseModel):
    """Record of a parameter change - logged for analysis.

    Every autonomous parameter adjustment is recorded for transparency
    and to enable rollback if needed.
    """
    evolution_id: str = Field(..., description="Unique evolution identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

    # What changed
    agent: str = Field(..., description="Agent name")
    parameter: str = Field(..., description="Parameter name")
    old_value: Any = Field(..., description="Previous value")
    new_value: Any = Field(..., description="New value")

    # Why it changed
    reason: str = Field(..., description="Reason for change")
    supporting_insights: List[str] = Field(
        default_factory=list,
        description="Insight IDs that suggested this change"
    )

    # Expected impact
    expected_impact: str = Field(
        default="", description="What improvement is expected"
    )

    # Validation (filled after observation period)
    validated: Optional[bool] = Field(None, description="Was the change beneficial?")
    performance_delta: Optional[float] = Field(
        None, description="Change in performance metric"
    )


class PositionReview(BaseModel):
    """Result of reflection agent reviewing an open position."""
    trade_id: str = Field(..., description="Trade being reviewed")
    symbol: str = Field(..., description="Trading symbol")
    review_timestamp: datetime = Field(default_factory=datetime.now)

    # Current state
    current_pnl: float = Field(..., description="Current unrealized P&L")
    current_pnl_pct: float = Field(..., description="Current return %")
    time_in_position: int = Field(..., description="Seconds since entry")

    # Regime comparison
    entry_regime: Dict[str, str] = Field(..., description="Regime at entry")
    current_regime: Dict[str, str] = Field(..., description="Current regime")
    regime_changed: bool = Field(default=False, description="Has regime changed?")

    # Recommendation
    action: PositionAction = Field(..., description="Suggested action")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in action")
    reason: str = Field(..., description="Reasoning for action")

    # Action details (if applicable)
    suggested_exit_price: Optional[float] = Field(None)
    suggested_stop: Optional[float] = Field(None)
    suggested_size_change: Optional[float] = Field(None)


class TradeScore(BaseModel):
    """Multi-objective score for a completed trade from Judge agent."""
    trade_id: str = Field(..., description="Trade being scored")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Overall score
    total_score: float = Field(..., ge=0, le=100, description="Total score 0-100")

    # Component scores (each 0-100)
    return_score: float = Field(..., description="Risk-adjusted return score")
    execution_score: float = Field(..., description="Slippage and fill quality")
    timing_score: float = Field(..., description="Entry/exit timing quality")
    risk_management_score: float = Field(..., description="Stop adherence, sizing")
    discipline_score: float = Field(..., description="Followed the plan")

    # Weights used
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "return": 0.35,
            "execution": 0.15,
            "timing": 0.15,
            "risk_management": 0.25,
            "discipline": 0.10,
        }
    )

    # Explanation
    explanation: str = Field(..., description="Detailed scoring explanation")


class DailyLearningReport(BaseModel):
    """Daily consolidation report summarizing learning progress."""
    report_id: str = Field(..., description="Report identifier")
    date: str = Field(..., description="Report date YYYY-MM-DD")
    generated_at: datetime = Field(default_factory=datetime.now)

    # Trade summary
    total_trades: int = Field(default=0)
    winning_trades: int = Field(default=0)
    losing_trades: int = Field(default=0)
    total_pnl: float = Field(default=0.0)

    # Performance by strategy
    strategy_performance: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance breakdown by strategy"
    )

    # Performance by regime
    regime_performance: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance breakdown by regime"
    )

    # New insights generated
    new_insights: List[str] = Field(
        default_factory=list,
        description="Insight IDs generated today"
    )

    # Parameter evolutions
    parameter_changes: List[str] = Field(
        default_factory=list,
        description="Evolution IDs from today"
    )

    # LLM summary
    summary: str = Field(default="", description="LLM-generated daily summary")
    key_learnings: List[str] = Field(
        default_factory=list,
        description="Key learnings from the day"
    )
