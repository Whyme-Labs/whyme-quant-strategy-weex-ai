"""Risk Manager Agent.

Evaluates and manages trading risk.
"""

import time
from typing import Any, Dict, Optional
from loguru import logger

from .base_agent import BaseAgent


class RiskManagerAgent(BaseAgent):
    """Manages trading risk and position sizing.

    This agent:
    - Evaluates proposed trade risk
    - Adjusts position sizes
    - Sets stop-loss and take-profit levels
    - Enforces risk limits
    """

    name = "risk_manager"
    stage_name = "Risk Assessment"
    model_name = "rule-based-risk-v1"

    def __init__(self, config: Dict[str, Any]):
        """Initialize risk manager.

        Args:
            config: Agent configuration
        """
        super().__init__(config)

        # Risk parameters
        self.max_position_pct = config.get("max_position_pct", 0.1)  # 10% of account
        self.max_leverage = config.get("max_leverage", 20)  # WEEX hackathon limit
        self.max_loss_per_trade_pct = config.get("max_loss_per_trade_pct", 0.02)  # 2%
        self.min_risk_reward = config.get("min_risk_reward", 1.5)

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trade risk.

        Args:
            context: Context with strategy_proposal and account info

        Returns:
            Risk assessment results
        """
        start_time = time.time()

        proposal = context.get("strategy_proposal", {})
        account = context.get("account", {})
        market_analysis = context.get("market_analysis", {})

        if not proposal or proposal.get("action") == "hold":
            return {
                "approved": False,
                "reason": "No trade proposed",
                "explanation": "No trade to evaluate",
                "confidence": 1.0,
            }

        assessment = await self._assess_risk(proposal, account, market_analysis)

        duration_ms = int((time.time() - start_time) * 1000)
        assessment["duration_ms"] = duration_ms

        return assessment

    async def _assess_risk(
        self,
        proposal: Dict[str, Any],
        account: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess trade risk.

        Args:
            proposal: Proposed trade
            account: Account information
            analysis: Market analysis

        Returns:
            Risk assessment
        """
        # Extract proposal details
        size = proposal.get("size", 0)
        price = proposal.get("price", 0)
        leverage = proposal.get("leverage", 1)

        # Get account info
        balance = account.get("available_balance", 10000)  # Default for testing
        current_positions = account.get("position_value", 0)

        # Calculate metrics
        trade_value = size * price if price else size
        position_pct = trade_value / balance if balance > 0 else 1

        # Risk checks
        checks = []
        approved = True

        # Check 1: Position size
        if position_pct > self.max_position_pct:
            checks.append(f"Position too large: {position_pct*100:.1f}% > {self.max_position_pct*100:.1f}%")
            approved = False

        # Check 2: Leverage limit (WEEX hackathon: max 20x)
        if leverage > self.max_leverage:
            checks.append(f"Leverage too high: {leverage}x > {self.max_leverage}x")
            approved = False

        # Check 3: Trend alignment
        trend = analysis.get("trend", "neutral")
        action = proposal.get("action", "").lower()

        if action == "buy" and trend == "bearish":
            checks.append("Warning: Buying against bearish trend")
        elif action == "sell" and trend == "bullish":
            checks.append("Warning: Selling against bullish trend")

        # Calculate adjusted size if needed
        adjusted_size = size
        if position_pct > self.max_position_pct:
            adjusted_size = (self.max_position_pct * balance) / price if price else size * 0.5

        # Calculate confidence
        confidence = 0.9
        if checks:
            confidence -= len(checks) * 0.1

        # Build explanation
        if approved:
            explanation = f"Trade approved. Position: {position_pct*100:.1f}% of balance. Leverage: {leverage}x."
        else:
            explanation = f"Trade rejected: {'; '.join(checks)}"

        # Calculate stop-loss and take-profit
        support = analysis.get("support", price * 0.95)
        resistance = analysis.get("resistance", price * 1.05)

        if action == "buy":
            stop_loss = support * 0.99
            take_profit = resistance * 1.01
        else:
            stop_loss = resistance * 1.01
            take_profit = support * 0.99

        return {
            "approved": approved,
            "checks": checks,
            "original_size": size,
            "adjusted_size": adjusted_size if adjusted_size != size else None,
            "position_pct": position_pct,
            "leverage": min(leverage, self.max_leverage),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": abs(take_profit - price) / abs(price - stop_loss) if abs(price - stop_loss) > 0 else 0,
            "confidence": max(confidence, 0.5),
            "explanation": explanation,
        }

    def get_input_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get summarized input for logging."""
        proposal = context.get("strategy_proposal", {})
        return {
            "action": proposal.get("action"),
            "size": proposal.get("size"),
            "price": proposal.get("price"),
            "leverage": proposal.get("leverage"),
        }
