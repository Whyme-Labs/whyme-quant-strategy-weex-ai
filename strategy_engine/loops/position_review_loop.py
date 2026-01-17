"""Position Review Loop.

Reviews open positions every hour (configurable) using the ReflectionAgent.
Delegates execution of actions to the ExecutorAgent (single point of execution).

Integrates with:
- WeexClient: Fetch positions
- ReflectionAgent: Review positions
- ExecutorAgent: Execute close/reduce/add orders (single point of execution)
- TradeMemoryService: Get trade records
- DiscordNotifier: Send notifications
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

from ..models.memory import PositionAction, PositionReview, TradeRecord, ExitReason
from ..services.trade_memory import TradeMemoryService
from ..agents.reflection_agent import ReflectionAgent


class PositionReviewLoop:
    """Async loop that reviews open positions periodically.

    Runs the ReflectionAgent on all open positions and delegates
    execution of suggested actions to the ExecutorAgent.

    Architecture:
    - ReflectionAgent: Read-only analyst that reviews positions
    - ExecutorAgent: Single point of execution for all trades
    """

    def __init__(
        self,
        weex_client,
        reflection_agent: ReflectionAgent,
        trade_memory: TradeMemoryService,
        executor_agent=None,
        regime_detector=None,
        discord=None,
        interval_seconds: int = 3600,  # 1 hour default
        symbol: str = "BTCUSDT",
    ):
        """Initialize Position Review Loop.

        Args:
            weex_client: WEEX API client (for fetching positions)
            reflection_agent: ReflectionAgent for position review
            trade_memory: TradeMemoryService for trade records
            executor_agent: ExecutorAgent for executing trades (single point of execution)
            regime_detector: Optional regime detector agent
            discord: Optional Discord notifier
            interval_seconds: Review interval in seconds
            symbol: Trading symbol
        """
        self.weex_client = weex_client
        self.reflection_agent = reflection_agent
        self.trade_memory = trade_memory
        self.executor_agent = executor_agent
        self.regime_detector = regime_detector
        self.discord = discord
        self.interval_seconds = interval_seconds
        self.symbol = symbol

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_review: Optional[datetime] = None

    async def start(self):
        """Start the position review loop."""
        if self._running:
            logger.warning("Position review loop already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Position review loop started (interval: {self.interval_seconds}s)")

    async def stop(self):
        """Stop the position review loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Position review loop stopped")

    async def _loop(self):
        """Main loop that runs reviews periodically."""
        while self._running:
            try:
                await self._review_cycle()
                self._last_review = datetime.now()

                # Wait for next interval
                await asyncio.sleep(self.interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position review loop: {e}")
                # Back off on error
                await asyncio.sleep(60)

    async def _review_cycle(self):
        """Run a single review cycle."""
        logger.info("Starting position review cycle")

        try:
            # 1. Fetch open positions from exchange
            positions = await self._fetch_positions()

            if not positions:
                logger.debug("No open positions to review")
                return

            logger.info(f"Reviewing {len(positions)} open position(s)")

            # 2. Get trade records for each position
            trade_records = await self._get_trade_records(positions)

            # 3. Get current market regime
            current_regime = await self._get_current_regime()

            # 4. Get current market data
            market_data = await self._get_market_data()

            # 5. Run reflection agent
            context = {
                "positions": positions,
                "trade_records": trade_records,
                "current_regime": current_regime,
                "market_data": market_data,
            }

            result = await self.reflection_agent.process(context)

            # 6. Execute actions for positions that need them
            actions_needed = result.get("actions_needed", [])

            if actions_needed:
                logger.info(f"Actions needed for {len(actions_needed)} position(s)")
                await self._execute_actions(actions_needed, positions, trade_records)

                # Notify Discord
                if self.discord:
                    await self._notify_reviews(actions_needed, result.get("explanation", ""))

            logger.info(f"Position review completed: {result.get('explanation', 'OK')}")

        except Exception as e:
            logger.error(f"Position review cycle failed: {e}")
            if self.discord:
                await self.discord.send_error(str(e), "Position Review Loop")

    async def _fetch_positions(self) -> List[Dict[str, Any]]:
        """Fetch open positions from exchange.

        Returns:
            List of position dictionaries
        """
        try:
            # Get all positions from WEEX API
            response = await self.weex_client.get_positions(self.symbol)

            if not response:
                return []

            # Handle different response formats
            if isinstance(response, list):
                positions = response
            elif isinstance(response, dict):
                positions = [response] if response.get("total") else []
            else:
                positions = []

            # Filter for non-zero positions
            return [p for p in positions if float(p.get("total", 0)) != 0]

        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def _get_trade_records(
        self,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, TradeRecord]:
        """Get trade records for positions.

        Args:
            positions: List of positions

        Returns:
            Dictionary mapping trade_id to TradeRecord
        """
        trade_records = {}

        # Get open trades from memory
        open_trades = await self.trade_memory.get_open_trades()

        for trade in open_trades:
            trade_records[trade.trade_id] = trade
            # Also map by order_id if available
            if trade.entry_order_id:
                trade_records[trade.entry_order_id] = trade

        return trade_records

    async def _get_current_regime(self) -> Dict[str, str]:
        """Get current market regime.

        Returns:
            Regime dictionary
        """
        if not self.regime_detector:
            return {"volatility": "normal", "trend": "unknown", "volume": "normal"}

        try:
            # Get market data for regime detection
            market_data = await self._get_market_data()
            result = await self.regime_detector.process({"market_data": market_data})
            return result.get("regime", {})
        except Exception as e:
            logger.error(f"Failed to get regime: {e}")
            return {"volatility": "unknown", "trend": "unknown", "volume": "unknown"}

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data.

        Returns:
            Market data dictionary
        """
        try:
            ticker = await self.weex_client.get_ticker(self.symbol)
            return {
                "symbol": self.symbol,
                "price": float(ticker.get("last", 0)),
                "bid": float(ticker.get("bestBid", 0)),
                "ask": float(ticker.get("bestAsk", 0)),
                "volume": float(ticker.get("baseVolume", 0)),
                "timestamp": ticker.get("timestamp"),
            }
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return {"symbol": self.symbol, "price": 0}

    async def _execute_actions(
        self,
        reviews: List[PositionReview],
        positions: List[Dict[str, Any]],
        trade_records: Dict[str, TradeRecord],
    ):
        """Execute suggested actions from reviews.

        Args:
            reviews: List of position reviews with actions
            positions: Current positions
            trade_records: Trade record mapping
        """
        for review in reviews:
            try:
                action = review.action
                trade_id = review.trade_id

                logger.info(
                    f"Executing {action.value} for {review.symbol} "
                    f"(confidence: {review.confidence:.0%})"
                )

                if action == PositionAction.CLOSE:
                    await self._close_position(review, positions)

                elif action == PositionAction.REDUCE:
                    await self._reduce_position(review, positions)

                elif action == PositionAction.ADD:
                    await self._add_to_position(review, positions)

                elif action == PositionAction.ADJUST_STOP:
                    await self._adjust_stop(review, positions)

            except Exception as e:
                logger.error(f"Failed to execute {action.value} for {review.symbol}: {e}")

    async def _close_position(
        self,
        review: PositionReview,
        positions: List[Dict[str, Any]],
    ):
        """Close a position via ExecutorAgent.

        Args:
            review: Position review
            positions: Current positions
        """
        # Find the matching position
        position = next(
            (p for p in positions if p.get("symbol", "").upper().replace("CMT_", "") == review.symbol.upper()),
            None
        )

        if not position:
            logger.warning(f"Position not found for {review.symbol}")
            return

        size = float(position.get("total", 0))
        if size <= 0:
            return

        # Delegate to ExecutorAgent (single point of execution)
        if self.executor_agent:
            result = await self.executor_agent.execute_close(
                trade_id=review.trade_id,
                position=position,
                reason=ExitReason.REFLECTION,
                regime=review.current_regime,
                exit_price=review.suggested_exit_price or float(position.get("markPrice", 0)),
            )
            if result and result.get("success"):
                logger.info(f"Position {review.symbol} closed via ExecutorAgent - {review.reason}")
            else:
                logger.error(f"ExecutorAgent failed to close position {review.symbol}")
        else:
            logger.error("No ExecutorAgent available - cannot close position")

    async def _reduce_position(
        self,
        review: PositionReview,
        positions: List[Dict[str, Any]],
    ):
        """Reduce a position via ExecutorAgent.

        Args:
            review: Position review
            positions: Current positions
        """
        position = next(
            (p for p in positions if p.get("symbol", "").upper().replace("CMT_", "") == review.symbol.upper()),
            None
        )

        if not position:
            return

        current_size = float(position.get("total", 0))
        if current_size <= 0:
            return

        # Calculate reduce percentage (default 50% if not specified)
        reduce_pct = abs(review.suggested_size_change or 0.5)

        # Delegate to ExecutorAgent (single point of execution)
        if self.executor_agent:
            result = await self.executor_agent.execute_reduce(
                trade_id=review.trade_id,
                position=position,
                reduce_pct=reduce_pct,
                reason=review.reason,
            )
            if result and result.get("success"):
                logger.info(f"Position {review.symbol} reduced by {reduce_pct*100:.0f}% via ExecutorAgent")
            else:
                logger.error(f"ExecutorAgent failed to reduce position {review.symbol}")
        else:
            logger.error("No ExecutorAgent available - cannot reduce position")

    async def _add_to_position(
        self,
        review: PositionReview,
        positions: List[Dict[str, Any]],
    ):
        """Add to an existing position (pyramid) via ExecutorAgent.

        Args:
            review: Position review
            positions: Current positions
        """
        position = next(
            (p for p in positions if p.get("symbol", "").upper().replace("CMT_", "") == review.symbol.upper()),
            None
        )

        if not position:
            return

        current_size = float(position.get("total", 0))
        if current_size <= 0:
            return

        # Calculate add percentage (default 50% of current position)
        add_pct = abs(review.suggested_size_change or 0.5)

        # Delegate to ExecutorAgent (single point of execution)
        if self.executor_agent:
            result = await self.executor_agent.execute_add(
                trade_id=review.trade_id,
                position=position,
                add_pct=add_pct,
                reason=review.reason,
            )
            if result and result.get("success"):
                logger.info(f"Position {review.symbol} increased by {add_pct*100:.0f}% via ExecutorAgent")
            else:
                logger.error(f"ExecutorAgent failed to add to position {review.symbol}")
        else:
            logger.error("No ExecutorAgent available - cannot add to position")

    async def _adjust_stop(
        self,
        review: PositionReview,
        positions: List[Dict[str, Any]],
    ):
        """Adjust stop loss for a position.

        Args:
            review: Position review
            positions: Current positions
        """
        if not review.suggested_stop:
            return

        # Note: Stop loss adjustment depends on WEEX API capabilities
        # This is a placeholder - actual implementation depends on
        # whether WEEX supports modifying stop orders

        logger.info(
            f"Would adjust stop for {review.symbol} to {review.suggested_stop:.2f} "
            f"(not implemented in WEEX API)"
        )

        # For now, log the suggestion for manual review
        if self.discord:
            await self.discord.send_trace(
                "Stop Adjustment Suggested",
                f"Consider adjusting stop for {review.symbol}",
                {
                    "Symbol": review.symbol,
                    "New Stop": f"${review.suggested_stop:.2f}",
                    "Reason": review.reason,
                }
            )

    async def _notify_reviews(
        self,
        reviews: List[PositionReview],
        summary: str,
    ):
        """Send Discord notifications for reviews.

        Args:
            reviews: Position reviews with actions
            summary: Review summary
        """
        if not self.discord:
            return

        for review in reviews:
            color = {
                PositionAction.CLOSE: 0xFF6B6B,  # Red
                PositionAction.REDUCE: 0xFFB347,  # Orange
                PositionAction.ADD: 0x77DD77,    # Green
                PositionAction.ADJUST_STOP: 0x6495ED,  # Blue
                PositionAction.HOLD: 0x808080,   # Gray
            }.get(review.action, 0x808080)

            await self.discord.send_status(
                f"Position Review: {review.action.value.upper()}",
                f"**Symbol:** {review.symbol}\n"
                f"**Action:** {review.action.value}\n"
                f"**Confidence:** {review.confidence*100:.0f}%\n"
                f"**Reason:** {review.reason}\n"
                f"**Current P&L:** {review.current_pnl_pct:+.2f}%\n"
                f"**Time in Position:** {review.time_in_position // 3600}h",
                color=color,
            )

    async def force_review(self) -> Dict[str, Any]:
        """Force an immediate position review.

        Returns:
            Review results
        """
        await self._review_cycle()
        return {"status": "completed", "last_review": self._last_review}
