"""Consolidation Loop.

Daily learning consolidation that:
1. Gets all trades from the lookback period
2. Runs Learner agent for pattern extraction
3. Generates insights and stores in semantic memory
4. Proposes and applies parameter evolutions
5. Generates daily report

Runs daily at a configurable time (default: 00:00 UTC).
"""

import asyncio
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional
from loguru import logger

from ..models.memory import (
    TradeRecord,
    StrategyInsight,
    ParameterEvolution,
    ParameterState,
    DailyLearningReport,
)
from ..services.trade_memory import TradeMemoryService
from ..agents.learner_agent import LearnerAgent


class ConsolidationLoop:
    """Async loop for daily learning consolidation.

    Runs the Learner agent to extract patterns, generate insights,
    and evolve parameters based on recent trading performance.
    """

    def __init__(
        self,
        trade_memory: TradeMemoryService,
        learner_agent: LearnerAgent,
        discord=None,
        trade_journal=None,
        run_time: time = time(0, 0),  # 00:00 UTC default
        lookback_days: int = 30,
        enable_auto_evolution: bool = True,
    ):
        """Initialize Consolidation Loop.

        Args:
            trade_memory: TradeMemoryService for data access
            learner_agent: LearnerAgent for pattern extraction
            discord: Discord notifier
            trade_journal: TradeJournal for human-readable summaries
            run_time: Time of day to run (UTC)
            lookback_days: Days of history to analyze
            enable_auto_evolution: Whether to auto-apply parameter changes
        """
        self.trade_memory = trade_memory
        self.learner_agent = learner_agent
        self.discord = discord
        self.trade_journal = trade_journal
        self.run_time = run_time
        self.lookback_days = lookback_days
        self.enable_auto_evolution = enable_auto_evolution

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_consolidation: Optional[datetime] = None
        self._last_report: Optional[DailyLearningReport] = None

    async def start(self):
        """Start the consolidation loop."""
        if self._running:
            logger.warning("Consolidation loop already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Consolidation loop started (runs daily at {self.run_time})")

    async def stop(self):
        """Stop the consolidation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Consolidation loop stopped")

    async def _loop(self):
        """Main loop that runs consolidation daily."""
        while self._running:
            try:
                # Calculate time until next run
                now = datetime.utcnow()
                next_run = datetime.combine(now.date(), self.run_time)

                # If we've passed today's run time, schedule for tomorrow
                if now.time() >= self.run_time:
                    next_run += timedelta(days=1)

                # Calculate seconds until next run
                seconds_until_run = (next_run - now).total_seconds()

                logger.info(f"Next consolidation in {seconds_until_run / 3600:.1f} hours")

                # Wait until run time
                await asyncio.sleep(seconds_until_run)

                # Run consolidation
                if self._running:  # Check if still running after sleep
                    await self._consolidation_cycle()
                    self._last_consolidation = datetime.utcnow()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation loop: {e}")
                # Wait an hour before retrying
                await asyncio.sleep(3600)

    async def _consolidation_cycle(self):
        """Run a single consolidation cycle."""
        logger.info("Starting daily consolidation cycle")

        try:
            # 1. Get trades from lookback period
            since = datetime.now() - timedelta(days=self.lookback_days)
            trades = await self.trade_memory.get_trades_since(since)

            if not trades:
                logger.info("No trades in lookback period - skipping consolidation")
                return

            logger.info(f"Analyzing {len(trades)} trades from last {self.lookback_days} days")

            # 2. Get current parameters
            current_parameters = await self._get_current_parameters()

            # 3. Run learner agent
            context = {
                "trades": trades,
                "current_parameters": current_parameters,
            }

            result = await self.learner_agent.consolidate_learning(trades, current_parameters)

            insights = result.get("insights", [])
            evolutions = result.get("evolutions", [])
            report = result.get("report")

            logger.info(
                f"Consolidation complete: {len(insights)} insights, "
                f"{len(evolutions)} parameter changes"
            )

            # 4. Store insights in semantic memory and notify Discord
            for insight in insights:
                await self.trade_memory.store_insight(insight)
                # Send individual insight notification
                if self.discord:
                    await self.discord.send_insight(
                        pattern=insight.pattern,
                        description=insight.description,
                        confidence=insight.confidence,
                        recommendation=insight.recommendation,
                        applies_when=insight.applies_when,
                        sample_size=insight.sample_size,
                    )

            # 5. Apply parameter evolutions (if enabled)
            if self.enable_auto_evolution and evolutions:
                applied = await self.learner_agent.apply_evolutions(
                    evolutions,
                    self.trade_memory,
                )
                logger.info(f"Applied {len(applied)} parameter evolutions")

                # Notify Discord about each parameter evolution
                if self.discord:
                    for evo in applied:
                        await self.discord.send_parameter_evolution(
                            agent=evo.agent,
                            parameter=evo.parameter,
                            old_value=evo.old_value,
                            new_value=evo.new_value,
                            reason=evo.reason,
                        )

            # 6. Store daily report
            if report:
                await self.trade_memory.store_daily_report(report)
                self._last_report = report

            # 7. Send Discord notification
            await self._notify_consolidation(trades, insights, evolutions, report)

        except Exception as e:
            logger.error(f"Consolidation cycle failed: {e}")
            if self.discord:
                await self.discord.send_error(str(e), "Daily Consolidation")

    async def _get_current_parameters(self) -> Dict[str, Dict[str, ParameterState]]:
        """Get current parameter states for all agents.

        Returns:
            Dictionary of agent -> parameter -> ParameterState
        """
        parameters = {}

        # List of agents with tunable parameters
        agents = [
            "portfolio_manager",
            "mean_reversion",
            "trend_following",
            "turtle_trading",
            "risk_manager",
        ]

        for agent in agents:
            params = await self.trade_memory.get_all_parameters(agent)
            if params:
                parameters[agent] = params

        return parameters

    async def _notify_consolidation(
        self,
        trades: List[TradeRecord],
        insights: List[StrategyInsight],
        evolutions: List[ParameterEvolution],
        report: Optional[DailyLearningReport],
    ):
        """Send Discord notification for consolidation results.

        Args:
            trades: Analyzed trades
            insights: Extracted insights
            evolutions: Parameter evolutions
            report: Daily report
        """
        # Send trade journal daily summary (professional format)
        if self.trade_journal:
            try:
                await self.trade_journal.send_daily_summary()
            except Exception as e:
                logger.warning(f"Trade journal daily summary failed: {e}")

        if not self.discord:
            return

        # Calculate summary stats
        from ..models.memory import TradeStatus
        closed = [t for t in trades if t.status == TradeStatus.CLOSED]
        wins = [t for t in closed if t.pnl and t.pnl > 0]
        total_pnl = sum(t.pnl for t in closed if t.pnl)

        # Format insights
        insights_text = ""
        for i, insight in enumerate(insights[:3], 1):
            insights_text += f"{i}. **{insight.pattern}**\n   {insight.description[:100]}...\n"
        if not insights_text:
            insights_text = "No new patterns detected"

        # Format evolutions
        evolutions_text = ""
        for evo in evolutions[:3]:
            evolutions_text += f"- {evo.agent}.{evo.parameter}: {evo.old_value} → {evo.new_value}\n"
        if not evolutions_text:
            evolutions_text = "No parameter changes"

        # Determine color based on P&L
        color = 0x2ECC71 if total_pnl >= 0 else 0xE74C3C

        await self.discord.send_status(
            "Daily Learning Report",
            f"**Period:** Last {self.lookback_days} days\n"
            f"**Trades Analyzed:** {len(closed)}\n"
            f"**Win Rate:** {len(wins)/len(closed)*100:.1f}%\n" if closed else "N/A\n"
            f"**Total P&L:** ${total_pnl:.2f}\n\n"
            f"**New Insights:**\n{insights_text}\n"
            f"**Parameter Evolution:**\n{evolutions_text}",
            color=color,
        )

        # Send key learnings if available
        if report and report.key_learnings:
            learnings_text = "\n".join(f"• {l}" for l in report.key_learnings)
            await self.discord.send_status(
                "Key Learnings",
                learnings_text,
                color=0x6495ED,
            )

    async def force_consolidation(self) -> Dict[str, Any]:
        """Force an immediate consolidation.

        Returns:
            Consolidation results
        """
        await self._consolidation_cycle()
        return {
            "status": "completed",
            "last_consolidation": self._last_consolidation,
            "last_report": self._last_report,
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get consolidation loop status.

        Returns:
            Status dictionary
        """
        now = datetime.utcnow()
        next_run = datetime.combine(now.date(), self.run_time)
        if now.time() >= self.run_time:
            next_run += timedelta(days=1)

        return {
            "running": self._running,
            "last_consolidation": self._last_consolidation,
            "next_consolidation": next_run,
            "lookback_days": self.lookback_days,
            "auto_evolution_enabled": self.enable_auto_evolution,
        }

    def get_last_report(self) -> Optional[DailyLearningReport]:
        """Get the most recent consolidation report.

        Returns:
            Last DailyLearningReport or None
        """
        return self._last_report


class TriggerBasedConsolidation:
    """Event-triggered consolidation for significant events.

    Runs consolidation when:
    - Large loss occurs (>5% drawdown)
    - Win streak ends (3+ consecutive wins followed by loss)
    - Unusual market regime detected
    """

    def __init__(
        self,
        consolidation_loop: ConsolidationLoop,
        min_interval_hours: int = 4,
    ):
        """Initialize trigger-based consolidation.

        Args:
            consolidation_loop: Main consolidation loop
            min_interval_hours: Minimum hours between triggered runs
        """
        self.consolidation_loop = consolidation_loop
        self.min_interval = timedelta(hours=min_interval_hours)
        self._last_triggered: Optional[datetime] = None
        self._consecutive_wins = 0

    async def on_trade_closed(self, trade: TradeRecord):
        """Check if consolidation should be triggered.

        Args:
            trade: Just-closed trade
        """
        # Check minimum interval
        if self._last_triggered:
            if datetime.now() - self._last_triggered < self.min_interval:
                return

        should_trigger = False
        reason = ""

        # Check for large loss
        if trade.pnl_pct and trade.pnl_pct < -5:
            should_trigger = True
            reason = f"Large loss: {trade.pnl_pct:.2f}%"

        # Track win streaks
        if trade.pnl and trade.pnl > 0:
            self._consecutive_wins += 1
        else:
            if self._consecutive_wins >= 3:
                should_trigger = True
                reason = f"Win streak ended after {self._consecutive_wins} wins"
            self._consecutive_wins = 0

        # Trigger if needed
        if should_trigger:
            logger.info(f"Triggered consolidation: {reason}")
            self._last_triggered = datetime.now()
            await self.consolidation_loop.force_consolidation()
