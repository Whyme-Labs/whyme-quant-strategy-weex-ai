"""Portfolio Manager Agent.

The critical bridge between strategy signals and execution.
Determines whether to execute trades based on:
1. Current portfolio positions and exposure
2. Market conditions and regime
3. Dynamic confidence thresholds
4. Correlation with existing positions
5. Risk budget allocation

"A signal is just a suggestion. Portfolio context determines execution."
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_RISK_ASSESSMENT


class ExecutionDecision(Enum):
    """Final execution decision."""
    EXECUTE = "execute"
    REDUCE_SIZE = "reduce_size"
    REJECT = "reject"
    QUEUE = "queue"  # Wait for better conditions


@dataclass
class Position:
    """Current portfolio position."""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


@dataclass
class PortfolioState:
    """Current portfolio state."""
    positions: List[Position] = field(default_factory=list)
    total_equity: float = 0.0
    used_margin: float = 0.0
    available_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    win_rate_recent: float = 0.5  # Last N trades
    max_drawdown_today: float = 0.0


@dataclass
class ExecutionPlan:
    """Final execution plan after portfolio analysis."""
    decision: ExecutionDecision
    original_signal: Dict[str, Any]
    adjusted_size: float
    adjusted_stop_price: Optional[float]
    adjusted_target_price: Optional[float]
    confidence_threshold_used: float
    signal_confidence: float
    reasoning: str
    portfolio_impact: Dict[str, Any]


class PortfolioManagerAgent(BaseAgent):
    """Portfolio Manager - The execution gatekeeper.

    This agent sits between strategy signals and actual execution.
    It makes the final call on whether to trade based on portfolio context.

    Key Responsibilities:
    1. Track all current positions
    2. Calculate portfolio-level metrics (exposure, correlation, drawdown)
    3. Dynamically adjust confidence thresholds
    4. Approve/reject/modify trade proposals
    5. Manage risk budget allocation across strategies

    Philosophy:
    - A good signal in a bad portfolio context should be rejected
    - Confidence thresholds should adapt to market conditions
    - Correlation matters - don't pile into correlated positions
    - Preserve capital first, maximize returns second
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize portfolio manager.

        Args:
            config: Configuration with:
                - max_portfolio_exposure: Max total exposure (default 0.5 = 50%)
                - max_single_position: Max single position size (default 0.1 = 10%)
                - max_correlated_exposure: Max exposure to correlated assets (default 0.3)
                - base_confidence_threshold: Starting confidence threshold (default 0.6)
                - max_daily_drawdown: Max daily drawdown before stopping (default 0.05)
                - win_rate_lookback: Number of trades to calculate win rate (default 20)
        """
        super().__init__(config)
        self.max_portfolio_exposure = config.get("max_portfolio_exposure", 0.5)
        self.max_single_position = config.get("max_single_position", 0.1)
        self.max_correlated_exposure = config.get("max_correlated_exposure", 0.3)
        self.base_confidence_threshold = config.get("base_confidence_threshold", 0.6)
        self.max_daily_drawdown = config.get("max_daily_drawdown", 0.05)
        self.win_rate_lookback = config.get("win_rate_lookback", 20)

        # Portfolio state
        self.portfolio = PortfolioState()
        self.trade_history: List[Dict[str, Any]] = []

        # Correlation groups (simplified)
        self.correlation_groups = {
            "btc_related": ["BTCUSDT", "BTCUSD"],
            "eth_related": ["ETHUSDT", "ETHUSD"],
            "major_alts": ["SOLUSDT", "AVAXUSDT", "DOTUSDT"],
        }

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trade signal through portfolio lens.

        Args:
            context: Dictionary containing:
                - strategy_proposal: Signal from strategy agent
                - market_data: Current market data
                - regime: Market regime
                - account_info: Optional account information

        Returns:
            Execution plan with final decision
        """
        proposal = context.get("strategy_proposal", {})
        market_data = context.get("market_data", {})
        regime = context.get("regime", {})
        account_info = context.get("account_info", {})

        # Update portfolio state if account info provided
        if account_info:
            self._update_portfolio_state(account_info)

        # If no proposal or hold, skip
        if not proposal or proposal.get("action") == "hold":
            return {
                "decision": ExecutionDecision.REJECT.value,
                "reasoning": "No active signal to evaluate",
            }

        # Calculate dynamic confidence threshold
        confidence_threshold = self._calculate_dynamic_threshold(regime)

        # Check signal confidence against threshold
        signal_confidence = proposal.get("confidence", 0.5)

        # Evaluate portfolio impact
        impact = self._evaluate_portfolio_impact(proposal)

        # Make final decision
        plan = self._make_execution_decision(
            proposal=proposal,
            signal_confidence=signal_confidence,
            confidence_threshold=confidence_threshold,
            impact=impact,
            regime=regime,
        )

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_RISK_ASSESSMENT,
            model="portfolio_manager_v1",
            input_data={
                "signal": proposal,
                "signal_confidence": signal_confidence,
                "portfolio_exposure": self._get_current_exposure(),
                "regime": regime,
            },
            output_data={
                "decision": plan.decision.value,
                "confidence_threshold": confidence_threshold,
                "adjusted_size": plan.adjusted_size,
                "portfolio_impact": plan.portfolio_impact,
            },
            explanation=plan.reasoning,
        )

        return {
            "decision": plan.decision.value,
            "approved": plan.decision in [ExecutionDecision.EXECUTE, ExecutionDecision.REDUCE_SIZE],
            "adjusted_size": plan.adjusted_size,
            "adjusted_stop_price": plan.adjusted_stop_price,
            "adjusted_target_price": plan.adjusted_target_price,
            "confidence_threshold": confidence_threshold,
            "signal_confidence": signal_confidence,
            "reasoning": plan.reasoning,
            "portfolio_impact": plan.portfolio_impact,
        }

    def _calculate_dynamic_threshold(self, regime: Dict[str, Any]) -> float:
        """Calculate dynamic confidence threshold based on conditions.

        Threshold increases (more conservative) when:
        - High volatility
        - Near max drawdown
        - Low recent win rate
        - High portfolio exposure

        Threshold decreases (more aggressive) when:
        - Low volatility with clear trend
        - Strong recent performance
        - Low portfolio exposure

        Args:
            regime: Current market regime

        Returns:
            Adjusted confidence threshold (0.0 - 1.0)
        """
        threshold = self.base_confidence_threshold

        # Volatility adjustment
        volatility = regime.get("volatility", "medium")
        if volatility == "high":
            threshold += 0.1  # More conservative in high vol
        elif volatility == "low":
            threshold -= 0.05  # Slightly more aggressive in low vol

        # Drawdown adjustment
        if self.portfolio.max_drawdown_today > self.max_daily_drawdown * 0.5:
            threshold += 0.15  # Much more conservative near drawdown limit
        elif self.portfolio.max_drawdown_today > self.max_daily_drawdown * 0.3:
            threshold += 0.08

        # Win rate adjustment
        if self.portfolio.win_rate_recent < 0.4:
            threshold += 0.1  # More conservative on losing streak
        elif self.portfolio.win_rate_recent > 0.6:
            threshold -= 0.05  # Slightly more aggressive on winning streak

        # Exposure adjustment
        current_exposure = self._get_current_exposure()
        if current_exposure > self.max_portfolio_exposure * 0.7:
            threshold += 0.1  # More selective when already exposed

        # Clamp to valid range
        return max(0.3, min(0.9, threshold))

    def _evaluate_portfolio_impact(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how the proposed trade impacts portfolio.

        Args:
            proposal: Trade proposal

        Returns:
            Impact analysis
        """
        symbol = proposal.get("symbol", "BTCUSDT")
        action = proposal.get("action", "hold")
        size = proposal.get("size", 0)

        # Current exposure
        current_exposure = self._get_current_exposure()
        new_exposure = current_exposure + (size / (self.portfolio.total_equity or 1))

        # Check correlation with existing positions
        correlated_exposure = self._get_correlated_exposure(symbol)

        # Check if this is adding to or reducing a position
        existing_position = self._get_position(symbol)
        is_adding = existing_position and (
            (existing_position.side == "long" and action == "buy") or
            (existing_position.side == "short" and action == "sell")
        )

        # Check if this is a hedge (opposite direction to existing)
        is_hedge = existing_position and not is_adding

        return {
            "current_exposure": current_exposure,
            "new_exposure": new_exposure,
            "correlated_exposure": correlated_exposure,
            "is_adding_to_position": is_adding,
            "is_hedge": is_hedge,
            "exposure_increase": new_exposure - current_exposure,
            "exceeds_max_exposure": new_exposure > self.max_portfolio_exposure,
            "exceeds_correlation_limit": correlated_exposure > self.max_correlated_exposure,
        }

    def _make_execution_decision(
        self,
        proposal: Dict[str, Any],
        signal_confidence: float,
        confidence_threshold: float,
        impact: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> ExecutionPlan:
        """Make final execution decision.

        Args:
            proposal: Trade proposal
            signal_confidence: Signal's confidence
            confidence_threshold: Dynamic threshold
            impact: Portfolio impact analysis
            regime: Market regime

        Returns:
            ExecutionPlan with final decision
        """
        reasoning_parts = []
        original_size = proposal.get("size", 0)
        adjusted_size = original_size

        # Get original TP/SL
        entry_price = proposal.get("price", 0)
        original_stop = proposal.get("stop_price")
        original_target = proposal.get("target_price")
        adjusted_stop = original_stop
        adjusted_target = original_target

        # Rule 1: Confidence check
        if signal_confidence < confidence_threshold:
            reasoning_parts.append(
                f"Signal confidence ({signal_confidence:.2f}) below threshold ({confidence_threshold:.2f})"
            )
            return ExecutionPlan(
                decision=ExecutionDecision.REJECT,
                original_signal=proposal,
                adjusted_size=0,
                adjusted_stop_price=None,
                adjusted_target_price=None,
                confidence_threshold_used=confidence_threshold,
                signal_confidence=signal_confidence,
                reasoning=". ".join(reasoning_parts) + ". REJECTED: Insufficient confidence.",
                portfolio_impact=impact,
            )

        # Rule 2: Max exposure check
        if impact["exceeds_max_exposure"]:
            # Can we reduce size to fit?
            max_allowed = (self.max_portfolio_exposure - impact["current_exposure"]) * (self.portfolio.total_equity or 1)
            if max_allowed > original_size * 0.3:  # At least 30% of original
                adjusted_size = max_allowed
                reasoning_parts.append(
                    f"Reduced size from {original_size:.2f} to {adjusted_size:.2f} to stay within exposure limit"
                )
            else:
                reasoning_parts.append(
                    f"Would exceed max portfolio exposure ({self.max_portfolio_exposure:.0%})"
                )
                return ExecutionPlan(
                    decision=ExecutionDecision.REJECT,
                    original_signal=proposal,
                    adjusted_size=0,
                    adjusted_stop_price=None,
                    adjusted_target_price=None,
                    confidence_threshold_used=confidence_threshold,
                    signal_confidence=signal_confidence,
                    reasoning=". ".join(reasoning_parts) + ". REJECTED: Exposure limit.",
                    portfolio_impact=impact,
                )

        # Rule 3: Correlation check
        if impact["exceeds_correlation_limit"] and not impact["is_hedge"]:
            # Reduce size for correlated positions
            adjusted_size = adjusted_size * 0.5
            reasoning_parts.append(
                f"Reduced size by 50% due to correlation with existing positions"
            )

        # Rule 4: Drawdown protection
        if self.portfolio.max_drawdown_today > self.max_daily_drawdown:
            reasoning_parts.append(
                f"Daily drawdown limit ({self.max_daily_drawdown:.1%}) reached"
            )
            return ExecutionPlan(
                decision=ExecutionDecision.REJECT,
                original_signal=proposal,
                adjusted_size=0,
                adjusted_stop_price=None,
                adjusted_target_price=None,
                confidence_threshold_used=confidence_threshold,
                signal_confidence=signal_confidence,
                reasoning=". ".join(reasoning_parts) + ". REJECTED: Drawdown limit hit.",
                portfolio_impact=impact,
            )

        # Rule 5: Regime-specific adjustments
        if regime.get("volatility") == "high":
            adjusted_size = adjusted_size * 0.7
            reasoning_parts.append("Reduced size by 30% due to high volatility")

            # Widen stop loss in high volatility (add 20% buffer)
            if adjusted_stop and entry_price:
                action = proposal.get("action", "buy")
                if action == "buy":
                    # For long, stop is below entry
                    stop_distance = entry_price - adjusted_stop
                    adjusted_stop = entry_price - (stop_distance * 1.2)
                else:
                    # For short, stop is above entry
                    stop_distance = adjusted_stop - entry_price
                    adjusted_stop = entry_price + (stop_distance * 1.2)
                reasoning_parts.append("Widened SL by 20% for high volatility")

        # Rule 6: Adjust TP/SL based on confidence
        if entry_price and signal_confidence > 0.8:
            # High confidence - can use tighter stop and wider target
            if adjusted_target:
                target_distance = abs(adjusted_target - entry_price)
                action = proposal.get("action", "buy")
                if action == "buy":
                    adjusted_target = entry_price + (target_distance * 1.15)
                else:
                    adjusted_target = entry_price - (target_distance * 1.15)
                reasoning_parts.append("Extended TP by 15% due to high confidence")

        # Rule 7: Ensure minimum risk/reward ratio of 1.5:1
        if adjusted_stop and adjusted_target and entry_price:
            risk = abs(entry_price - adjusted_stop)
            reward = abs(adjusted_target - entry_price)
            if risk > 0 and reward / risk < 1.5:
                # Adjust target to meet minimum R:R
                action = proposal.get("action", "buy")
                min_reward = risk * 1.5
                if action == "buy":
                    adjusted_target = entry_price + min_reward
                else:
                    adjusted_target = entry_price - min_reward
                reasoning_parts.append(f"Adjusted TP to maintain 1.5:1 R:R ratio")

        # All checks passed
        reasoning_parts.append(
            f"Confidence {signal_confidence:.2f} > threshold {confidence_threshold:.2f}"
        )

        # Determine final decision
        if adjusted_size < original_size * 0.5:
            decision = ExecutionDecision.REDUCE_SIZE
            reasoning_parts.append(f"Executing with reduced size: {adjusted_size:.2f}")
        else:
            decision = ExecutionDecision.EXECUTE
            reasoning_parts.append(f"Executing with size: {adjusted_size:.2f}")

        return ExecutionPlan(
            decision=decision,
            original_signal=proposal,
            adjusted_size=adjusted_size,
            adjusted_stop_price=adjusted_stop,
            adjusted_target_price=adjusted_target,
            confidence_threshold_used=confidence_threshold,
            signal_confidence=signal_confidence,
            reasoning=". ".join(reasoning_parts),
            portfolio_impact=impact,
        )

    def _get_current_exposure(self) -> float:
        """Get current portfolio exposure as fraction of equity."""
        if not self.portfolio.total_equity:
            return 0.0
        return self.portfolio.used_margin / self.portfolio.total_equity

    def _get_correlated_exposure(self, symbol: str) -> float:
        """Get exposure to assets correlated with given symbol."""
        # Find which correlation group this symbol belongs to
        symbol_group = None
        for group_name, symbols in self.correlation_groups.items():
            if symbol in symbols:
                symbol_group = group_name
                break

        if not symbol_group:
            return 0.0

        # Sum exposure to all symbols in the same group
        correlated_exposure = 0.0
        for position in self.portfolio.positions:
            if position.symbol in self.correlation_groups.get(symbol_group, []):
                correlated_exposure += position.margin_used

        if self.portfolio.total_equity:
            return correlated_exposure / self.portfolio.total_equity
        return 0.0

    def _get_position(self, symbol: str) -> Optional[Position]:
        """Get existing position for symbol."""
        for position in self.portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    def _update_portfolio_state(self, account_info: Dict[str, Any]):
        """Update portfolio state from account info.

        Args:
            account_info: Account information from exchange
        """
        self.portfolio.total_equity = float(account_info.get("equity", 0))
        self.portfolio.used_margin = float(account_info.get("usedMargin", 0))
        self.portfolio.available_margin = float(account_info.get("availableMargin", 0))
        self.portfolio.unrealized_pnl = float(account_info.get("unrealizedPnl", 0))

        # Update positions
        positions_data = account_info.get("positions", [])
        self.portfolio.positions = [
            Position(
                symbol=p.get("symbol", ""),
                side=p.get("side", "").lower(),
                size=float(p.get("size", 0)),
                entry_price=float(p.get("entryPrice", 0)),
                unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                margin_used=float(p.get("margin", 0)),
            )
            for p in positions_data
            if float(p.get("size", 0)) > 0
        ]

    def record_trade_result(self, trade: Dict[str, Any]):
        """Record a completed trade for win rate calculation.

        Args:
            trade: Completed trade with pnl
        """
        self.trade_history.append(trade)

        # Keep only recent trades
        if len(self.trade_history) > self.win_rate_lookback:
            self.trade_history = self.trade_history[-self.win_rate_lookback:]

        # Update win rate
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
            self.portfolio.win_rate_recent = wins / len(self.trade_history)

    def update_daily_drawdown(self, current_equity: float, starting_equity: float):
        """Update daily drawdown calculation.

        Args:
            current_equity: Current account equity
            starting_equity: Equity at start of day
        """
        if starting_equity > 0:
            drawdown = (starting_equity - current_equity) / starting_equity
            self.portfolio.max_drawdown_today = max(
                self.portfolio.max_drawdown_today,
                drawdown
            )
