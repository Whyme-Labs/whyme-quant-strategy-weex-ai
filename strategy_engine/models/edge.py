"""Edge Models for Statistical Edge Collection System.

This module defines the core data models for the edge-based trading approach:
- Edge: A registered statistical edge with measurable properties
- EdgeSignal: A signal generated when edge conditions are met
- EdgeHealth: Health status of an edge
- PositionSize: Result of Kelly-based position sizing

Philosophy: "Quantification is not about predicting future prices, but
systematically collecting probability advantages in local market inefficiencies."
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field


class EdgeStatus(str, Enum):
    """Status of an edge in the registry."""
    ACTIVE = "active"       # Trading enabled
    WARMING = "warming"     # Collecting data, not trading yet
    PAUSED = "paused"       # Temporarily disabled (manual or degrading)
    DISABLED = "disabled"   # Permanently disabled (negative edge proven)


class EdgeType(str, Enum):
    """Type of edge strategy."""
    # Core strategies
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    TURTLE = "turtle"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"

    # Indicator-based edges
    MACD = "macd"
    STOCHASTIC = "stochastic"
    ICHIMOKU = "ichimoku"
    SUPERTREND = "supertrend"

    # Volume-based edges
    VOLUME = "volume"

    # Pattern-based edges
    CHART_PATTERN = "chart_pattern"
    CANDLESTICK = "candlestick"


class EdgeHealthStatus(str, Enum):
    """Health status from monitoring."""
    HEALTHY = "healthy"         # Performing as expected
    DEGRADING = "degrading"     # Below expected but still positive
    BROKEN = "broken"           # Negative expectancy


class Edge(BaseModel):
    """A registered statistical edge with measurable properties.

    An edge represents a specific market pattern/condition that has historically
    provided a statistical advantage. Each edge tracks its own win rate, payoff
    ratio, and expected value to enable Kelly-based position sizing.
    """
    edge_id: str = Field(..., description="Unique edge identifier (e.g., 'mr_rsi_oversold_btc_4h')")
    name: str = Field(..., description="Human readable name")
    description: str = Field(default="", description="Description of the edge")
    edge_type: EdgeType = Field(..., description="Type of edge strategy")

    # Conditions that define this edge
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTCUSDT') or '*' for all")
    timeframe: str = Field(..., description="Primary timeframe (e.g., '1h', '4h', '1d')")
    entry_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for entry: {rsi_below: 30, bb_position: 'below_lower'}"
    )
    exit_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conditions for exit (or use standard stops)"
    )

    # Statistical properties (updated after each trade)
    sample_size: int = Field(default=0, description="Number of completed trades")
    wins: int = Field(default=0, description="Number of winning trades")
    losses: int = Field(default=0, description="Number of losing trades")

    # Win/loss metrics
    avg_win_pct: float = Field(default=0.0, description="Average winning trade return %")
    avg_loss_pct: float = Field(default=0.0, description="Average losing trade loss % (positive)")
    total_win_pct: float = Field(default=0.0, description="Sum of all winning trade returns")
    total_loss_pct: float = Field(default=0.0, description="Sum of all losing trade losses")

    # Derived statistics (for quick access)
    largest_win_pct: float = Field(default=0.0, description="Largest winning trade %")
    largest_loss_pct: float = Field(default=0.0, description="Largest losing trade %")
    win_streak: int = Field(default=0, description="Current win streak")
    loss_streak: int = Field(default=0, description="Current loss streak")
    max_win_streak: int = Field(default=0, description="Maximum win streak")
    max_loss_streak: int = Field(default=0, description="Maximum loss streak")

    # Sharpe-like metrics
    returns_std: float = Field(default=0.0, description="Standard deviation of returns")

    # Rolling performance (last 30 days)
    rolling_30d_trades: int = Field(default=0, description="Trades in last 30 days")
    rolling_30d_wins: int = Field(default=0, description="Wins in last 30 days")
    rolling_30d_total_return: float = Field(default=0.0, description="Total return % in last 30 days")

    # Health and status
    status: EdgeStatus = Field(default=EdgeStatus.WARMING, description="Current edge status")
    pause_reason: Optional[str] = Field(None, description="Reason if paused")

    # Minimum requirements to become active
    min_sample_size: int = Field(default=30, description="Min trades before trusting edge")
    min_expectancy: float = Field(default=0.01, description="Min expectancy (1%) to trade")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = Field(None, description="Time of last trade")

    @computed_field
    @property
    def win_rate(self) -> float:
        """Calculate win rate from wins and sample size."""
        if self.sample_size == 0:
            return 0.0
        return self.wins / self.sample_size

    @computed_field
    @property
    def loss_rate(self) -> float:
        """Calculate loss rate."""
        return 1.0 - self.win_rate

    @computed_field
    @property
    def payoff_ratio(self) -> float:
        """Calculate payoff ratio (avg_win / avg_loss)."""
        if self.avg_loss_pct == 0:
            return 0.0
        return self.avg_win_pct / self.avg_loss_pct

    @computed_field
    @property
    def expectancy(self) -> float:
        """Calculate expectancy per trade.

        Formula: (win_rate * avg_win) - (loss_rate * avg_loss)
        Returns: Expected return % per trade
        """
        return (self.win_rate * self.avg_win_pct) - (self.loss_rate * self.avg_loss_pct)

    @computed_field
    @property
    def kelly_fraction(self) -> float:
        """Calculate Kelly optimal fraction.

        Formula: (win_rate * payoff - loss_rate) / payoff
        Returns: Optimal fraction of bankroll to bet (can be > 1)
        """
        if self.payoff_ratio == 0:
            return 0.0
        return (self.win_rate * self.payoff_ratio - self.loss_rate) / self.payoff_ratio

    @computed_field
    @property
    def rolling_30d_win_rate(self) -> float:
        """Win rate over last 30 days."""
        if self.rolling_30d_trades == 0:
            return 0.0
        return self.rolling_30d_wins / self.rolling_30d_trades

    @computed_field
    @property
    def rolling_30d_expectancy(self) -> float:
        """Approximate expectancy from rolling 30d data."""
        if self.rolling_30d_trades == 0:
            return 0.0
        return self.rolling_30d_total_return / self.rolling_30d_trades

    @computed_field
    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio (expectancy / std)."""
        if self.returns_std == 0:
            return 0.0
        return self.expectancy / self.returns_std

    @computed_field
    @property
    def is_mature(self) -> bool:
        """Whether edge has enough sample size to be trusted."""
        return self.sample_size >= self.min_sample_size

    @computed_field
    @property
    def is_profitable(self) -> bool:
        """Whether edge has positive expectancy above minimum."""
        return self.expectancy >= self.min_expectancy

    @computed_field
    @property
    def can_trade(self) -> bool:
        """Whether this edge can be traded."""
        return (
            self.status == EdgeStatus.ACTIVE and
            self.is_mature and
            self.is_profitable
        )

    class Config:
        use_enum_values = True


class EdgeSignal(BaseModel):
    """A signal generated when edge conditions are met.

    This is the output of scanning an edge against current market conditions.
    Contains all context needed for position sizing and execution.
    """
    signal_id: str = Field(..., description="Unique signal identifier")
    edge_id: str = Field(..., description="Edge that generated this signal")
    edge: Edge = Field(..., description="Full edge object for sizing")

    # Signal details
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="'long' or 'short'")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Market context at signal
    price: float = Field(..., description="Price when signal generated")
    market_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market data snapshot: {price, bid, ask, volume, indicators}"
    )

    # Entry parameters (from edge conditions)
    suggested_entry: float = Field(..., description="Suggested entry price")
    suggested_stop: float = Field(..., description="Suggested stop loss")
    suggested_take_profit: Optional[float] = Field(None, description="Suggested take profit")

    # Risk metrics
    risk_pct: float = Field(default=0.0, description="Risk % from entry to stop")

    # Status
    executed: bool = Field(default=False, description="Whether signal was executed")
    execution_result: Optional[Dict[str, Any]] = Field(None, description="Execution result")

    class Config:
        use_enum_values = True


class PositionSize(BaseModel):
    """Result of Kelly-based position sizing calculation."""
    size: float = Field(..., description="Position size in base units")
    size_usd: float = Field(default=0.0, description="Position size in USD")
    position_pct: float = Field(default=0.0, description="Position as % of equity")

    # Kelly calculations
    kelly_raw: float = Field(default=0.0, description="Raw Kelly fraction")
    kelly_fractional: float = Field(default=0.0, description="Fractional Kelly used")

    # Edge metrics used
    edge_expectancy: float = Field(default=0.0, description="Edge expectancy")
    edge_win_rate: float = Field(default=0.0, description="Edge win rate")
    edge_payoff: float = Field(default=0.0, description="Edge payoff ratio")

    # Adjustments applied
    volatility_adjustment: float = Field(default=1.0, description="Vol adjustment factor")
    correlation_adjustment: float = Field(default=1.0, description="Correlation adjustment")

    # Limits applied
    capped_by_max: bool = Field(default=False, description="Was capped by max position")
    capped_by_correlation: bool = Field(default=False, description="Was capped by correlation limit")

    # Reason
    reason: str = Field(default="", description="Explanation of sizing")
    can_trade: bool = Field(default=True, description="Whether size > 0 and tradeable")


class EdgeHealth(BaseModel):
    """Health assessment of an edge from monitoring."""
    edge_id: str = Field(..., description="Edge being assessed")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Current status
    status: EdgeHealthStatus = Field(..., description="Health status")
    status_reason: str = Field(default="", description="Reason for status")

    # Performance comparison
    historical_expectancy: float = Field(..., description="All-time expectancy")
    rolling_expectancy: float = Field(..., description="Rolling 30d expectancy")
    expectancy_ratio: float = Field(default=0.0, description="Rolling / Historical")

    # Win rate comparison
    historical_win_rate: float = Field(..., description="All-time win rate")
    rolling_win_rate: float = Field(..., description="Rolling 30d win rate")

    # Sample sizes
    total_trades: int = Field(default=0)
    rolling_trades: int = Field(default=0)

    # Recommendations
    should_pause: bool = Field(default=False, description="Should edge be paused?")
    should_disable: bool = Field(default=False, description="Should edge be disabled?")


class EdgeTradeAttribution(BaseModel):
    """Attribution of a trade to an edge for tracking."""
    trade_id: str = Field(..., description="Trade ID")
    edge_id: str = Field(..., description="Edge ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Expected vs actual
    expected_pnl: float = Field(..., description="Expected P&L based on edge expectancy")
    actual_pnl: Optional[float] = Field(None, description="Actual P&L (filled on close)")

    # Edge state at entry
    edge_expectancy_at_entry: float = Field(default=0.0)
    edge_win_rate_at_entry: float = Field(default=0.0)
    edge_sample_size_at_entry: int = Field(default=0)

    # Position sizing
    position_size: float = Field(default=0.0)
    kelly_used: float = Field(default=0.0)

    # Trade outcome
    is_win: Optional[bool] = Field(None, description="Whether trade was profitable")
    return_pct: Optional[float] = Field(None, description="Actual return %")


class ConvergenceReport(BaseModel):
    """Report on how well actual results converge to expected edge."""
    edge_id: str = Field(..., description="Edge being analyzed")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Sample info
    sample_size: int = Field(default=0, description="Number of trades")
    period_days: int = Field(default=30, description="Analysis period")

    # Cumulative comparison
    expected_pnl: float = Field(default=0.0, description="Cumulative expected P&L")
    actual_pnl: float = Field(default=0.0, description="Cumulative actual P&L")
    convergence_ratio: float = Field(default=0.0, description="Actual / Expected")

    # Statistical assessment
    is_converging: bool = Field(default=False, description="Within acceptable range")
    convergence_status: str = Field(default="", description="CONVERGING, OUTPERFORMING, UNDERPERFORMING")

    # Confidence interval
    expected_variance: float = Field(default=0.0, description="Expected variance given sample size")
    within_2_std: bool = Field(default=True, description="Within 2 standard deviations")

    # Recommendations
    assessment: str = Field(default="", description="Assessment summary")
