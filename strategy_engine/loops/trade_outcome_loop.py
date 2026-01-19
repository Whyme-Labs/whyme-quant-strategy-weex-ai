"""Trade Outcome Loop.

Records and analyzes trade outcomes when positions are closed.
Triggered by trade close events (not periodic).

Responsibilities:
1. Update trade record with exit data
2. Calculate P&L, duration, excursions
3. Run Judge agent to score the trade
4. Run LLM for reflection
5. Store lessons in episodic memory

Integrates with:
- TradeMemoryService: Update trade records
- JudgeAgent: Score trades
- LLMAnalyzer: Generate reflections
- DiscordNotifier: Send notifications
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from ..models.memory import (
    TradeRecord,
    TradeStatus,
    ExitReason,
    TradeScore,
)
from ..services.trade_memory import TradeMemoryService
from ..agents.judge_agent import JudgeAgent


class TradeOutcomeLoop:
    """Handles trade outcome recording and analysis.

    Called when a trade is closed (by TP, SL, manual, or reflection).
    Records the outcome, scores the trade, and generates reflection.
    """

    def __init__(
        self,
        trade_memory: TradeMemoryService,
        judge_agent: JudgeAgent,
        llm_analyzer=None,
        discord=None,
    ):
        """Initialize Trade Outcome Loop.

        Args:
            trade_memory: TradeMemoryService for persistence
            judge_agent: JudgeAgent for scoring
            llm_analyzer: LLM for reflection generation
            discord: Discord notifier
        """
        self.trade_memory = trade_memory
        self.judge_agent = judge_agent
        self.llm = llm_analyzer
        self.discord = discord

        # Track recent outcomes for batch analysis
        self._recent_outcomes: List[TradeRecord] = []
        self._max_recent = 50

    async def on_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        exit_regime: Optional[Dict[str, str]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradeRecord]:
        """Handle a trade being closed.

        This is the main entry point, called when a trade exits.

        Args:
            trade_id: ID of the closed trade
            exit_price: Exit price
            exit_reason: Why the trade was closed
            exit_regime: Market regime at exit
            additional_data: Extra data (MFE, MAE, etc.)

        Returns:
            Updated TradeRecord or None
        """
        logger.info(f"Processing trade outcome: {trade_id} ({exit_reason.value})")

        try:
            # 1. Update trade record with exit data
            trade = await self.trade_memory.record_trade_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                exit_regime=exit_regime,
            )

            if not trade:
                logger.warning(f"Trade not found in memory: {trade_id}")
                return None

            # 2. Update MFE/MAE if provided
            if additional_data:
                mfe = additional_data.get("max_favorable_excursion")
                mae = additional_data.get("max_adverse_excursion")
                if mfe is not None or mae is not None:
                    trade = await self.trade_memory.update_excursions(
                        trade_id=trade_id,
                        mfe=mfe or 0,
                        mae=mae or 0,
                    )

            # 3. Score the trade
            score = await self.judge_agent.score_trade(trade)
            logger.info(f"Trade {trade_id} scored: {score.total_score:.1f}/100")

            # 4. Generate LLM reflection
            reflection = await self._generate_reflection(trade, score)

            # 5. Extract lessons
            lessons = await self._extract_lessons(trade, reflection)

            # 6. Update trade with reflection
            trade = await self.trade_memory.add_reflection(
                trade_id=trade_id,
                reflection=reflection,
                score=score.total_score,
                lessons=lessons,
                score_breakdown={
                    "return": score.return_score,
                    "execution": score.execution_score,
                    "timing": score.timing_score,
                    "risk_management": score.risk_management_score,
                    "discipline": score.discipline_score,
                }
            )

            # 7. Add to recent outcomes
            self._recent_outcomes.append(trade)
            if len(self._recent_outcomes) > self._max_recent:
                self._recent_outcomes = self._recent_outcomes[-self._max_recent:]

            # 8. Send notifications
            await self._notify_outcome(trade, score, reflection, lessons)

            logger.info(
                f"Trade outcome processed: {trade_id} | "
                f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%) | "
                f"Score: {score.total_score:.1f}"
            )

            return trade

        except Exception as e:
            logger.error(f"Failed to process trade outcome: {e}")
            if self.discord:
                await self.discord.send_error(str(e), f"Trade Outcome: {trade_id}")
            return None

    async def _generate_reflection(
        self,
        trade: TradeRecord,
        score: TradeScore,
    ) -> str:
        """Generate LLM reflection on the trade.

        Args:
            trade: Completed trade
            score: Trade score

        Returns:
            Reflection text
        """
        if not self.llm:
            return self._generate_basic_reflection(trade, score)

        try:
            # Build reflection prompt
            prompt = f"""
Analyze this completed trade and provide a brief reflection:

**Trade Details:**
- Symbol: {trade.symbol}
- Side: {trade.entry_side.upper()}
- Entry: ${trade.entry_price:.2f} at {trade.entry_timestamp.strftime('%Y-%m-%d %H:%M')}
- Exit: ${trade.exit_price:.2f} at {trade.exit_timestamp.strftime('%Y-%m-%d %H:%M') if trade.exit_timestamp else 'N/A'}
- P&L: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)
- Duration: {trade.duration_seconds // 3600 if trade.duration_seconds else 0} hours
- Exit Reason: {trade.exit_reason.value if trade.exit_reason else 'unknown'}

**Strategy:** {trade.entry_strategy}
**Entry Reasoning:** {trade.entry_reasoning}

**Market Regime at Entry:**
- Volatility: {trade.entry_regime.get('volatility', 'unknown')}
- Trend: {trade.entry_regime.get('trend', 'unknown')}

**Market Regime at Exit:**
- Volatility: {trade.exit_regime.get('volatility', 'unknown') if trade.exit_regime else 'unknown'}
- Trend: {trade.exit_regime.get('trend', 'unknown') if trade.exit_regime else 'unknown'}

**Score:** {score.total_score:.1f}/100
- Return: {score.return_score:.1f}
- Timing: {score.timing_score:.1f}
- Risk Management: {score.risk_management_score:.1f}

Provide a 2-3 sentence reflection on:
1. What went well or poorly
2. One specific lesson to apply next time
"""

            return await self.llm.analyze_market(
                symbol=trade.symbol,
                price=trade.exit_price or trade.entry_price,
                regime=trade.exit_regime or trade.entry_regime,
                additional_context=prompt,
            )

        except Exception as e:
            logger.error(f"LLM reflection failed: {e}")
            return self._generate_basic_reflection(trade, score)

    def _generate_basic_reflection(
        self,
        trade: TradeRecord,
        score: TradeScore,
    ) -> str:
        """Generate basic reflection without LLM.

        Args:
            trade: Completed trade
            score: Trade score

        Returns:
            Basic reflection text
        """
        parts = []

        # Result summary
        if trade.pnl and trade.pnl > 0:
            parts.append(f"Winning trade with {trade.pnl_pct:+.2f}% return.")
        else:
            parts.append(f"Losing trade with {trade.pnl_pct:+.2f}% return.")

        # Score analysis
        if score.total_score >= 70:
            parts.append("Overall execution was good.")
        elif score.total_score >= 50:
            parts.append("Execution was acceptable but could improve.")
        else:
            parts.append("Trade execution needs improvement.")

        # Specific feedback
        lowest_component = min(
            [("return", score.return_score),
             ("timing", score.timing_score),
             ("risk_management", score.risk_management_score)],
            key=lambda x: x[1]
        )

        if lowest_component[1] < 50:
            parts.append(f"Focus on improving {lowest_component[0].replace('_', ' ')}.")

        return " ".join(parts)

    async def _extract_lessons(
        self,
        trade: TradeRecord,
        reflection: str,
    ) -> List[str]:
        """Extract actionable lessons from the trade.

        Args:
            trade: Completed trade
            reflection: LLM reflection

        Returns:
            List of lessons learned
        """
        lessons = []

        # Extract based on outcomes
        if trade.pnl_pct and trade.pnl_pct < -3:
            lessons.append("Consider tighter stop losses for large losses")

        if trade.duration_seconds and trade.duration_seconds > 7 * 24 * 3600:  # >7 days
            lessons.append("Long hold time - consider time-based exits")

        if trade.exit_reason == ExitReason.STOP_LOSS and trade.pnl_pct and trade.pnl_pct < -2:
            lessons.append("Stop loss triggered with significant loss - review entry criteria")

        if trade.exit_reason == ExitReason.REGIME_CHANGE:
            lessons.append("Regime change triggered exit - good risk management")

        # Strategy-specific lessons
        strategy = trade.entry_strategy
        if strategy == "mean_reversion" and trade.pnl_pct and trade.pnl_pct < -2:
            if trade.entry_regime.get("volatility") == "high":
                lessons.append("Mean reversion in high volatility is risky - reduce size or skip")

        if strategy == "trend_following" and trade.pnl_pct and trade.pnl_pct < -1:
            if trade.entry_regime.get("trend") == "ranging":
                lessons.append("Trend following in ranging market - wait for confirmation")

        # Cap at 3 lessons
        return lessons[:3] if lessons else ["Trade recorded for pattern analysis"]

    async def _notify_outcome(
        self,
        trade: TradeRecord,
        score: TradeScore,
        reflection: str,
        lessons: List[str],
    ):
        """Send Discord notification for trade outcome.

        Args:
            trade: Completed trade
            score: Trade score
            reflection: LLM reflection
            lessons: Extracted lessons
        """
        if not self.discord:
            return

        # Determine color based on outcome
        if trade.pnl and trade.pnl > 0:
            color = 0x2ECC71  # Green
            result = "WIN"
        else:
            color = 0xE74C3C  # Red
            result = "LOSS"

        # Format lessons
        lessons_text = "\n".join(f"- {l}" for l in lessons) if lessons else "None"

        await self.discord.send_status(
            f"Trade Outcome: {result}",
            f"**{trade.symbol}** ({trade.entry_side.upper()})\n\n"
            f"**P&L:** ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)\n"
            f"**Score:** {score.total_score:.1f}/100\n"
            f"**Duration:** {trade.duration_seconds // 3600 if trade.duration_seconds else 0}h\n"
            f"**Exit Reason:** {trade.exit_reason.value if trade.exit_reason else 'unknown'}\n\n"
            f"**Reflection:**\n{reflection[:300]}...\n\n"
            f"**Lessons:**\n{lessons_text}",
            color=color,
        )

    async def process_batch_outcomes(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[TradeRecord]:
        """Process multiple trade outcomes.

        Useful for batch processing historical trades.

        Args:
            trades: List of trade close data

        Returns:
            List of processed TradeRecords
        """
        processed = []

        for trade_data in trades:
            trade = await self.on_trade_closed(
                trade_id=trade_data["trade_id"],
                exit_price=trade_data["exit_price"],
                exit_reason=ExitReason(trade_data.get("exit_reason", "manual")),
                exit_regime=trade_data.get("exit_regime"),
                additional_data=trade_data.get("additional_data"),
            )
            if trade:
                processed.append(trade)

        return processed

    def get_recent_outcomes(self, limit: int = 20) -> List[TradeRecord]:
        """Get recent trade outcomes.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent TradeRecords
        """
        return self._recent_outcomes[-limit:]

    async def check_for_closed_positions(
        self,
        weex_client,
        symbol: str,
        known_positions: Dict[str, str],
    ) -> List[str]:
        """Check if any tracked positions have been closed.

        Args:
            weex_client: WEEX client
            symbol: Trading symbol
            known_positions: Dict of known trade_id -> status

        Returns:
            List of trade_ids that have closed
        """
        closed = []

        try:
            # Get current positions
            current = await weex_client.get_position(symbol)
            current_ids = set()

            if isinstance(current, list):
                for p in current:
                    if float(p.get("size", 0)) != 0:  # API returns 'size' not 'total'
                        # Try to find matching trade_id
                        order_id = p.get("orderId", p.get("id", ""))  # API returns 'id' not 'positionId'
                        current_ids.add(str(order_id))

            # Find positions that are no longer open
            for trade_id, status in known_positions.items():
                if status == "open" and trade_id not in current_ids:
                    closed.append(trade_id)

        except Exception as e:
            logger.error(f"Failed to check closed positions: {e}")

        return closed
