"""Executor Agent - Single Point of Execution.

This is the ONLY component that can execute transactions.
All other agents are read-only analysts that provide recommendations.

Responsibilities:
- Execute open position orders
- Execute close position orders
- Execute reduce/add position orders
- Log all executions to Discord
- Record trades in memory
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from loguru import logger

from .base_agent import BaseAgent
from ..models.memory import TradeRecord, TradeStatus, ExitReason


class ExecutorAgent(BaseAgent):
    """Single point of execution for all trades.

    This agent is the ONLY component that can call weex_client.place_order().
    All other agents are read-only analysts.

    Architecture:
    - Analysts (read-only): RegimeDetector, MeanReversion, TrendFollowing, etc.
    - Gatekeeper (read-only): PortfolioManager approves/rejects
    - Executor (this): Executes approved trades
    """

    name = "executor_agent"
    stage_name = "Trade Execution"
    model_name = "executor-v1"

    def __init__(
        self,
        config: Dict[str, Any],
        weex_client=None,
        trade_memory=None,
        discord=None,
        llm_analyzer=None,
    ):
        """Initialize Executor Agent.

        Args:
            config: Agent configuration
            weex_client: WEEX API client (required for execution)
            trade_memory: TradeMemoryService for recording trades
            discord: Discord notifier for logging
            llm_analyzer: LLM for trade analysis
        """
        super().__init__(config)

        self.weex_client = weex_client
        self.trade_memory = trade_memory
        self.discord = discord
        self.llm = llm_analyzer

        # Execution parameters
        self.slippage_tolerance = config.get("slippage_tolerance", 0.001)
        self.prefer_limit_orders = config.get("prefer_limit_orders", True)

        # Track order_id -> trade_id mapping
        self._order_to_trade: Dict[str, str] = {}

        # Execution stats
        self._stats = {
            "total_executions": 0,
            "successful_opens": 0,
            "successful_closes": 0,
            "failed_executions": 0,
        }

    def set_dependencies(
        self,
        weex_client=None,
        trade_memory=None,
        discord=None,
        llm_analyzer=None,
    ):
        """Set dependencies after initialization.

        Useful when dependencies aren't available at init time.
        """
        if weex_client:
            self.weex_client = weex_client
        if trade_memory:
            self.trade_memory = trade_memory
        if discord:
            self.discord = discord
        if llm_analyzer:
            self.llm = llm_analyzer

    # =========================================================================
    # MAIN EXECUTION METHODS
    # =========================================================================

    async def execute_open(
        self,
        signal: Any,
        market_data: Dict[str, Any],
        regime: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Execute an OPEN position order.

        This is the ONLY method that should open new positions.

        Args:
            signal: Trading signal (from orchestrator)
            market_data: Current market data
            regime: Current market regime

        Returns:
            Execution result with order_id and trade_id, or None on failure
        """
        if not self.weex_client:
            logger.error("EXECUTOR: Cannot execute - weex_client not set")
            return None

        self._stats["total_executions"] += 1
        start_time = time.time()

        try:
            entry_price = signal.price or market_data.get("price", 0)
            symbol = signal.symbol

            # Log intent
            await self._log_discord(
                "EXECUTOR: Opening Position",
                f"**{signal.action.value.upper()} {symbol}**\n"
                f"Price: ${entry_price:,.2f}\n"
                f"Size: {signal.size}\n"
                f"Strategy: {signal.strategy}\n"
                f"Confidence: {signal.confidence*100:.0f}%",
                color=0x3498DB,  # Blue - pending
            )

            # Execute order
            result = await self.weex_client.place_order(
                symbol=symbol,
                side=signal.action.value,
                order_type="limit" if signal.price else "market",
                size=str(signal.size),
                price=str(signal.price) if signal.price else None,
            )

            order_id = result.get("orderId")
            if not order_id:
                raise Exception("No order ID returned from exchange")

            logger.info(f"EXECUTOR: Order placed - {order_id}")

            # Record trade in memory
            trade_id = str(uuid.uuid4())
            if self.trade_memory:
                trade_record = TradeRecord(
                    trade_id=trade_id,
                    symbol=symbol,
                    entry_timestamp=datetime.now(),
                    entry_price=entry_price,
                    entry_side="long" if signal.action.value == "buy" else "short",
                    entry_size=signal.size,
                    entry_strategy=signal.strategy,
                    entry_regime=regime.copy() if regime else {},
                    entry_confidence=signal.confidence,
                    entry_reasoning=signal.reason or "",
                    market_data_at_entry=market_data.copy(),
                    agent_decisions=[],
                    entry_order_id=str(order_id),
                    status=TradeStatus.OPEN,
                )
                await self.trade_memory.record_trade_entry(trade_record)

            # Track mapping
            self._order_to_trade[str(order_id)] = trade_id

            # Get LLM analysis
            llm_analysis = ""
            if self.llm:
                llm_analysis = await self.llm.analyze_signal(
                    symbol=symbol,
                    price=entry_price,
                    direction=signal.action.value,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    reasoning=signal.reason,
                    regime=regime,
                )

            # Log success
            duration_ms = int((time.time() - start_time) * 1000)
            color = 0x2ECC71 if signal.action.value == "buy" else 0xE74C3C

            await self._log_discord(
                f"EXECUTOR: Position Opened ({signal.action.value.upper()})",
                f"**{symbol}** @ ${entry_price:,.2f}\n\n"
                f"**Order ID:** `{order_id}`\n"
                f"**Trade ID:** `{trade_id[:8]}...`\n"
                f"**Size:** {signal.size}\n"
                f"**Strategy:** {signal.strategy}\n"
                f"**TP:** ${signal.target_price:,.2f}" if signal.target_price else "Not set" + "\n"
                f"**SL:** ${signal.stop_price:,.2f}" if signal.stop_price else "Not set" + "\n"
                f"**Execution Time:** {duration_ms}ms\n\n"
                f"**Regime:** {regime.get('volatility', 'N/A')} vol, {regime.get('trend', 'N/A')} trend\n\n"
                f"**AI Analysis:**\n{llm_analysis[:500] if llm_analysis else 'N/A'}",
                color=color,
            )

            self._stats["successful_opens"] += 1

            return {
                "success": True,
                "order_id": str(order_id),
                "trade_id": trade_id,
                "entry_price": entry_price,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"EXECUTOR: Failed to open position - {e}")
            self._stats["failed_executions"] += 1

            await self._log_discord(
                "EXECUTOR: Open Position FAILED",
                f"**Error:** {str(e)}\n"
                f"**Symbol:** {signal.symbol}\n"
                f"**Direction:** {signal.action.value}",
                color=0xFF0000,  # Red
            )

            return None

    async def execute_close(
        self,
        trade_id: str,
        position: Dict[str, Any],
        reason: ExitReason,
        regime: Optional[Dict[str, str]] = None,
        exit_price: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a CLOSE position order.

        This is the ONLY method that should close positions.

        Args:
            trade_id: Trade ID to close
            position: Current position data from exchange
            reason: Reason for closing
            regime: Current market regime
            exit_price: Optional specific exit price

        Returns:
            Execution result or None on failure
        """
        if not self.weex_client:
            logger.error("EXECUTOR: Cannot close - weex_client not set")
            return None

        self._stats["total_executions"] += 1
        start_time = time.time()

        try:
            symbol = position.get("symbol", "").replace("cmt_", "").upper() or "BTCUSDT"
            hold_side = position.get("holdSide", "long")
            close_side = "sell" if hold_side == "long" else "buy"
            size = abs(float(position.get("total", 0)))

            if size <= 0:
                logger.warning(f"EXECUTOR: No position to close for {trade_id}")
                return None

            current_price = exit_price or float(position.get("markPrice", 0))

            # Log intent
            await self._log_discord(
                "EXECUTOR: Closing Position",
                f"**{symbol}** ({hold_side.upper()})\n"
                f"Size: {size}\n"
                f"Reason: {reason.value}\n"
                f"Exit Price: ~${current_price:,.2f}",
                color=0xF39C12,  # Orange - closing
            )

            # Execute close order
            result = await self.weex_client.place_order(
                symbol=symbol,
                side=close_side,
                order_type="market",
                size=str(size),
                trade_side="close",
            )

            order_id = result.get("orderId")
            logger.info(f"EXECUTOR: Close order placed - {order_id}")

            # Record exit in memory
            if self.trade_memory:
                await self.trade_memory.record_trade_exit(
                    trade_id=trade_id,
                    exit_price=current_price,
                    exit_reason=reason,
                    exit_regime=regime,
                )

            # Calculate P&L if we have entry data
            pnl_str = "Calculating..."
            if self.trade_memory:
                trade = await self.trade_memory.get_trade(trade_id)
                if trade and trade.pnl is not None:
                    pnl_str = f"${trade.pnl:,.2f} ({trade.pnl_pct:+.2f}%)"

            # Log success
            duration_ms = int((time.time() - start_time) * 1000)
            color = 0x2ECC71 if "profit" in reason.value.lower() else 0xE74C3C

            await self._log_discord(
                f"EXECUTOR: Position Closed ({reason.value})",
                f"**{symbol}** ({hold_side.upper()})\n\n"
                f"**Order ID:** `{order_id}`\n"
                f"**Trade ID:** `{trade_id[:8]}...`\n"
                f"**Exit Price:** ${current_price:,.2f}\n"
                f"**Size Closed:** {size}\n"
                f"**P&L:** {pnl_str}\n"
                f"**Execution Time:** {duration_ms}ms",
                color=color,
            )

            self._stats["successful_closes"] += 1

            return {
                "success": True,
                "order_id": str(order_id) if order_id else None,
                "trade_id": trade_id,
                "exit_price": current_price,
                "reason": reason.value,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"EXECUTOR: Failed to close position - {e}")
            self._stats["failed_executions"] += 1

            await self._log_discord(
                "EXECUTOR: Close Position FAILED",
                f"**Error:** {str(e)}\n"
                f"**Trade ID:** {trade_id}\n"
                f"**Reason:** {reason.value}",
                color=0xFF0000,
            )

            return None

    async def execute_reduce(
        self,
        trade_id: str,
        position: Dict[str, Any],
        reduce_pct: float = 0.5,
        reason: str = "Risk reduction",
    ) -> Optional[Dict[str, Any]]:
        """Execute a REDUCE position order.

        Args:
            trade_id: Trade ID to reduce
            position: Current position data
            reduce_pct: Percentage to reduce (0-1)
            reason: Reason for reduction

        Returns:
            Execution result or None on failure
        """
        if not self.weex_client:
            return None

        self._stats["total_executions"] += 1

        try:
            symbol = position.get("symbol", "").replace("cmt_", "").upper() or "BTCUSDT"
            hold_side = position.get("holdSide", "long")
            close_side = "sell" if hold_side == "long" else "buy"
            current_size = abs(float(position.get("total", 0)))
            reduce_size = current_size * reduce_pct

            if reduce_size <= 0:
                return None

            await self._log_discord(
                "EXECUTOR: Reducing Position",
                f"**{symbol}** ({hold_side.upper()})\n"
                f"Reducing by: {reduce_pct*100:.0f}%\n"
                f"Size: {reduce_size:.4f}\n"
                f"Reason: {reason}",
                color=0xF39C12,
            )

            result = await self.weex_client.place_order(
                symbol=symbol,
                side=close_side,
                order_type="market",
                size=str(reduce_size),
                trade_side="close",
            )

            order_id = result.get("orderId")

            await self._log_discord(
                "EXECUTOR: Position Reduced",
                f"**{symbol}** reduced by {reduce_pct*100:.0f}%\n"
                f"Order ID: `{order_id}`",
                color=0x9B59B6,
            )

            return {
                "success": True,
                "order_id": str(order_id) if order_id else None,
                "reduced_size": reduce_size,
            }

        except Exception as e:
            logger.error(f"EXECUTOR: Failed to reduce position - {e}")
            self._stats["failed_executions"] += 1
            return None

    async def execute_add(
        self,
        trade_id: str,
        position: Dict[str, Any],
        add_pct: float = 0.5,
        reason: str = "Pyramiding",
    ) -> Optional[Dict[str, Any]]:
        """Execute an ADD to position order (pyramid).

        Args:
            trade_id: Trade ID to add to
            position: Current position data
            add_pct: Percentage of current position to add
            reason: Reason for adding

        Returns:
            Execution result or None on failure
        """
        if not self.weex_client:
            return None

        self._stats["total_executions"] += 1

        try:
            symbol = position.get("symbol", "").replace("cmt_", "").upper() or "BTCUSDT"
            hold_side = position.get("holdSide", "long")
            add_side = "buy" if hold_side == "long" else "sell"
            current_size = abs(float(position.get("total", 0)))
            add_size = current_size * add_pct

            if add_size <= 0:
                return None

            await self._log_discord(
                "EXECUTOR: Adding to Position",
                f"**{symbol}** ({hold_side.upper()})\n"
                f"Adding: {add_pct*100:.0f}% ({add_size:.4f})\n"
                f"Reason: {reason}",
                color=0x3498DB,
            )

            result = await self.weex_client.place_order(
                symbol=symbol,
                side=add_side,
                order_type="market",
                size=str(add_size),
            )

            order_id = result.get("orderId")

            await self._log_discord(
                "EXECUTOR: Position Increased",
                f"**{symbol}** added {add_pct*100:.0f}%\n"
                f"Order ID: `{order_id}`",
                color=0x2ECC71,
            )

            return {
                "success": True,
                "order_id": str(order_id) if order_id else None,
                "added_size": add_size,
            }

        except Exception as e:
            logger.error(f"EXECUTOR: Failed to add to position - {e}")
            self._stats["failed_executions"] += 1
            return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def _log_discord(
        self,
        title: str,
        message: str,
        color: int = 0x3498DB,
    ):
        """Send a log message to Discord.

        Args:
            title: Message title
            message: Message body
            color: Embed color
        """
        if self.discord:
            try:
                await self.discord.send_status(title, message, color=color)
            except Exception as e:
                logger.error(f"Failed to send Discord log: {e}")

    def get_order_to_trade_mapping(self) -> Dict[str, str]:
        """Get the order_id -> trade_id mapping.

        Used by position monitor to track which trades to update on close.
        """
        return self._order_to_trade.copy()

    def get_trade_id_for_order(self, order_id: str) -> Optional[str]:
        """Get trade_id for a given order_id."""
        return self._order_to_trade.get(str(order_id))

    def remove_order_mapping(self, order_id: str):
        """Remove an order from tracking (after it's closed)."""
        if str(order_id) in self._order_to_trade:
            del self._order_to_trade[str(order_id)]

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._stats.copy()

    # =========================================================================
    # BASE AGENT INTERFACE (for compatibility with orchestrator)
    # =========================================================================

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process method for orchestrator compatibility.

        Note: This plans execution but doesn't execute.
        Actual execution happens through execute_open/execute_close.
        """
        proposal = context.get("strategy_proposal", {})
        risk = context.get("risk_assessment", {})
        market_data = context.get("market_data", {})

        if not risk.get("approved", False):
            return {
                "execute": False,
                "reason": "Not approved by risk manager",
                "explanation": "Trade not approved",
            }

        # Plan execution (but don't execute here)
        action = proposal.get("action", "").lower()
        size = risk.get("adjusted_size") or proposal.get("size", 0)
        current_price = market_data.get("price", 0)

        return {
            "execute": True,
            "order_type": "limit" if self.prefer_limit_orders else "market",
            "price": current_price,
            "size": size,
            "confidence": 0.85,
            "explanation": f"Execution planned: {action} {size} units",
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
        }
