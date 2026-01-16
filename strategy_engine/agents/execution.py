"""Execution Agent.

Optimizes order execution timing and method.
"""

import time
from typing import Any, Dict, Optional
from loguru import logger

from .base_agent import BaseAgent


class ExecutionAgent(BaseAgent):
    """Optimizes trade execution.

    This agent:
    - Determines optimal order type (market vs limit)
    - Calculates optimal entry price
    - Manages execution timing
    - Handles order splitting for large positions
    """

    name = "execution_agent"
    stage_name = "Execution Planning"
    model_name = "rule-based-exec-v1"

    def __init__(self, config: Dict[str, Any]):
        """Initialize execution agent.

        Args:
            config: Agent configuration
        """
        super().__init__(config)

        # Execution parameters
        self.slippage_tolerance = config.get("slippage_tolerance", 0.001)  # 0.1%
        self.prefer_limit_orders = config.get("prefer_limit_orders", True)
        self.split_threshold = config.get("split_threshold", 10000)  # Split orders > $10k

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan trade execution.

        Args:
            context: Context with strategy_proposal and risk_assessment

        Returns:
            Execution plan
        """
        start_time = time.time()

        proposal = context.get("strategy_proposal", {})
        risk = context.get("risk_assessment", {})
        market_data = context.get("market_data", {})

        if not risk.get("approved", False):
            return {
                "execute": False,
                "reason": "Not approved by risk manager",
                "explanation": "Trade not approved",
                "confidence": 1.0,
            }

        plan = await self._plan_execution(proposal, risk, market_data)

        duration_ms = int((time.time() - start_time) * 1000)
        plan["duration_ms"] = duration_ms

        return plan

    async def _plan_execution(
        self,
        proposal: Dict[str, Any],
        risk: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Plan execution strategy.

        Args:
            proposal: Trade proposal
            risk: Risk assessment
            market_data: Current market data

        Returns:
            Execution plan
        """
        action = proposal.get("action", "").lower()
        size = risk.get("adjusted_size") or proposal.get("size", 0)
        current_price = market_data.get("price", 0)
        bid = market_data.get("bid", current_price * 0.999)
        ask = market_data.get("ask", current_price * 1.001)
        spread = ask - bid if ask > bid else 0

        # Determine order type
        spread_pct = spread / current_price if current_price > 0 else 0

        if spread_pct > self.slippage_tolerance:
            # Wide spread - use limit order
            order_type = "limit"
            if action == "buy":
                price = bid + (spread * 0.3)  # Slightly above bid
            else:
                price = ask - (spread * 0.3)  # Slightly below ask
        else:
            # Tight spread - can use market
            if self.prefer_limit_orders:
                order_type = "limit"
                price = current_price
            else:
                order_type = "market"
                price = None

        # Check if order should be split
        trade_value = size * (price or current_price)
        split_orders = []

        if trade_value > self.split_threshold:
            # Split into smaller orders
            num_splits = int(trade_value / self.split_threshold) + 1
            split_size = size / num_splits
            for i in range(num_splits):
                split_orders.append({
                    "size": split_size,
                    "sequence": i + 1,
                })

        # Build explanation
        if order_type == "limit":
            explanation = f"Execute {action} with limit order at {price:.2f}. "
        else:
            explanation = f"Execute {action} with market order. "

        if split_orders:
            explanation += f"Split into {len(split_orders)} orders for better execution."
        else:
            explanation += "Single order execution."

        return {
            "execute": True,
            "order_type": order_type,
            "price": price,
            "size": size,
            "split_orders": split_orders if split_orders else None,
            "stop_loss": risk.get("stop_loss"),
            "take_profit": risk.get("take_profit"),
            "time_in_force": "GTC",  # Good till cancelled
            "confidence": 0.85,
            "explanation": explanation,
        }

    def get_input_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get summarized input for logging."""
        proposal = context.get("strategy_proposal", {})
        risk = context.get("risk_assessment", {})
        market = context.get("market_data", {})

        return {
            "action": proposal.get("action"),
            "size": risk.get("adjusted_size") or proposal.get("size"),
            "approved": risk.get("approved"),
            "current_price": market.get("price"),
            "spread": market.get("ask", 0) - market.get("bid", 0),
        }
