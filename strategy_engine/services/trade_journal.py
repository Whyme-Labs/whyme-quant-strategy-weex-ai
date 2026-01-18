"""Trade Journal Service - Human-readable trade logging.

Creates professional trading journal entries like a human trader would:
- Entry analysis: Why we entered, what signals triggered
- Exit analysis: Why we exited, what we learned
- Daily/weekly summaries
- Export to markdown for review
- Statistics and performance metrics
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from .redis_client import RedisClient
from .trade_memory import TradeMemoryService
from ..models.memory import TradeRecord, TradeStatus, ExitReason
from shared.discord import DiscordNotifier


class TradeJournalEntry:
    """A single trade journal entry."""

    def __init__(self, trade: TradeRecord):
        self.trade = trade

    def to_markdown(self) -> str:
        """Convert to markdown format for human review."""
        trade = self.trade
        lines = []

        # Header
        direction = trade.entry_side.upper()
        status_icon = self._get_status_icon()
        lines.append(f"## {status_icon} {direction} {trade.symbol} - {trade.trade_id[:8]}")
        lines.append("")

        # Entry Section
        lines.append("### Entry")
        lines.append(f"- **Time:** {trade.entry_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"- **Price:** ${trade.entry_price:,.2f}")
        lines.append(f"- **Size:** {trade.entry_size:.4f}")
        lines.append(f"- **Strategy:** {trade.entry_strategy}")
        lines.append(f"- **Confidence:** {trade.entry_confidence*100:.0f}%")
        lines.append("")

        # Market Context
        if trade.entry_regime:
            lines.append("### Market Context at Entry")
            for key, value in trade.entry_regime.items():
                lines.append(f"- **{key.title()}:** {value}")
            lines.append("")

        # Entry Reasoning
        lines.append("### Why I Entered")
        lines.append(f"> {trade.entry_reasoning}")
        lines.append("")

        # Exit Section (if closed)
        if trade.status == TradeStatus.CLOSED and trade.exit_timestamp:
            lines.append("### Exit")
            lines.append(f"- **Time:** {trade.exit_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append(f"- **Price:** ${trade.exit_price:,.2f}")
            lines.append(f"- **Reason:** {self._format_exit_reason(trade.exit_reason)}")
            lines.append("")

            # Duration
            if trade.duration_seconds:
                duration = self._format_duration(trade.duration_seconds)
                lines.append(f"- **Duration:** {duration}")
                lines.append("")

            # P&L
            lines.append("### Result")
            pnl_icon = "+" if (trade.pnl or 0) >= 0 else ""
            pnl_color = "green" if (trade.pnl or 0) >= 0 else "red"
            lines.append(f"- **P&L:** {pnl_icon}${trade.pnl or 0:,.2f} ({pnl_icon}{(trade.pnl_pct or 0)*100:.2f}%)")

            if trade.max_favorable_excursion is not None:
                lines.append(f"- **Max Favorable (MFE):** ${trade.max_favorable_excursion:,.2f}")
            if trade.max_adverse_excursion is not None:
                lines.append(f"- **Max Adverse (MAE):** ${trade.max_adverse_excursion:,.2f}")
            lines.append("")

            # Score
            if trade.score is not None:
                lines.append("### Trade Score")
                lines.append(f"- **Overall:** {trade.score:.0f}/100")
                if trade.score_breakdown:
                    for category, score in trade.score_breakdown.items():
                        lines.append(f"  - {category}: {score:.0f}")
                lines.append("")

            # Reflection
            if trade.reflection:
                lines.append("### Reflection")
                lines.append(f"> {trade.reflection}")
                lines.append("")

            # Lessons Learned
            if trade.lessons_learned:
                lines.append("### Lessons Learned")
                for lesson in trade.lessons_learned:
                    lines.append(f"- {lesson}")
                lines.append("")

        # Separator
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def to_discord_embed(self) -> Dict[str, Any]:
        """Convert to Discord embed format."""
        trade = self.trade
        direction = trade.entry_side.upper()

        # Build fields
        fields = [
            {"name": "Entry Price", "value": f"${trade.entry_price:,.2f}", "inline": True},
            {"name": "Size", "value": f"{trade.entry_size:.4f}", "inline": True},
            {"name": "Confidence", "value": f"{trade.entry_confidence*100:.0f}%", "inline": True},
            {"name": "Strategy", "value": trade.entry_strategy, "inline": True},
        ]

        if trade.entry_regime:
            regime_str = ", ".join(f"{k}: {v}" for k, v in trade.entry_regime.items())
            fields.append({"name": "Regime", "value": regime_str, "inline": False})

        # Add exit info if closed
        if trade.status == TradeStatus.CLOSED:
            fields.extend([
                {"name": "Exit Price", "value": f"${trade.exit_price:,.2f}", "inline": True},
                {"name": "P&L", "value": f"${trade.pnl or 0:+,.2f} ({(trade.pnl_pct or 0)*100:+.2f}%)", "inline": True},
                {"name": "Exit Reason", "value": self._format_exit_reason(trade.exit_reason), "inline": True},
            ])

            if trade.score is not None:
                fields.append({"name": "Score", "value": f"{trade.score:.0f}/100", "inline": True})

        # Color based on status/result
        if trade.status == TradeStatus.CLOSED:
            color = 0x00FF00 if (trade.pnl or 0) >= 0 else 0xFF6B6B
        elif trade.status == TradeStatus.OPEN:
            color = 0x00D4FF
        else:
            color = 0x808080

        return {
            "title": f"{self._get_status_icon()} {direction} {trade.symbol}",
            "description": trade.entry_reasoning[:200] + "..." if len(trade.entry_reasoning) > 200 else trade.entry_reasoning,
            "color": color,
            "fields": fields,
            "footer": {"text": f"Trade ID: {trade.trade_id[:8]}"},
            "timestamp": trade.entry_timestamp.isoformat(),
        }

    def _get_status_icon(self) -> str:
        """Get status icon for trade."""
        trade = self.trade
        if trade.status == TradeStatus.CLOSED:
            if (trade.pnl or 0) > 0:
                return "W"  # Win
            elif (trade.pnl or 0) < 0:
                return "L"  # Loss
            else:
                return "="  # Break-even
        elif trade.status == TradeStatus.OPEN:
            return "O"  # Open
        elif trade.status == TradeStatus.PENDING:
            return "?"  # Pending
        else:
            return "X"  # Cancelled

    def _format_exit_reason(self, reason: Optional[ExitReason]) -> str:
        """Format exit reason for display."""
        if reason is None:
            return "Unknown"

        reason_map = {
            ExitReason.TAKE_PROFIT: "Take Profit Hit",
            ExitReason.STOP_LOSS: "Stop Loss Hit",
            ExitReason.TRAILING_STOP: "Trailing Stop",
            ExitReason.REGIME_CHANGE: "Regime Change",
            ExitReason.THESIS_BROKEN: "Thesis Broken",
            ExitReason.MANUAL: "Manual Close",
            ExitReason.REFLECTION: "AI Reflection",
            ExitReason.TIMEOUT: "Time Limit",
            ExitReason.LIQUIDATION: "Liquidation",
        }
        return reason_map.get(reason, str(reason))

    def _format_duration(self, seconds: int) -> str:
        """Format duration for display."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"


class TradeJournal:
    """Trade Journal Service - Creates professional trading journal entries."""

    REDIS_PREFIX = "journal"

    def __init__(
        self,
        trade_memory: TradeMemoryService,
        redis_client: RedisClient,
        discord: Optional[DiscordNotifier] = None,
    ):
        """Initialize Trade Journal.

        Args:
            trade_memory: Trade memory service for accessing trades
            redis_client: Redis client for journal storage
            discord: Discord notifier for sending journal updates
        """
        self.trade_memory = trade_memory
        self.redis = redis_client
        self.discord = discord

    async def create_entry(self, trade: TradeRecord) -> TradeJournalEntry:
        """Create a journal entry for a trade.

        Args:
            trade: TradeRecord to journal

        Returns:
            TradeJournalEntry object
        """
        entry = TradeJournalEntry(trade)

        # Store in Redis
        if self.redis and self.redis.is_connected:
            key = f"{self.REDIS_PREFIX}:entry:{trade.trade_id}"
            await self.redis.set(key, entry.to_markdown())

            # Add to daily index
            date_str = trade.entry_timestamp.strftime("%Y-%m-%d")
            index_key = f"{self.REDIS_PREFIX}:daily:{date_str}"
            await self.redis.client.sadd(index_key, trade.trade_id)

        logger.info(f"Created journal entry for trade {trade.trade_id[:8]}")
        return entry

    async def send_entry_notification(self, trade: TradeRecord):
        """Send trade entry notification to Discord.

        Args:
            trade: TradeRecord for the entry
        """
        if not self.discord:
            return

        entry = TradeJournalEntry(trade)
        embed = entry.to_discord_embed()

        await self.discord.send_custom_embed(
            title=f"Trade Journal: {embed['title']}",
            description=embed['description'],
            color=embed['color'],
            fields=embed['fields'],
        )

    async def send_exit_notification(self, trade: TradeRecord):
        """Send trade exit notification to Discord with full journal entry.

        Args:
            trade: TradeRecord (completed) for the exit
        """
        if not self.discord:
            return

        entry = TradeJournalEntry(trade)

        # For exits, include more detail
        pnl_icon = "+" if (trade.pnl or 0) >= 0 else ""
        result = "WIN" if (trade.pnl or 0) > 0 else "LOSS" if (trade.pnl or 0) < 0 else "BREAK-EVEN"

        await self.discord.send_custom_embed(
            title=f"Trade Closed: {result} on {trade.symbol}",
            description=f"**P&L:** {pnl_icon}${trade.pnl or 0:,.2f} ({pnl_icon}{(trade.pnl_pct or 0)*100:.2f}%)\n\n**Exit Reason:** {entry._format_exit_reason(trade.exit_reason)}",
            color=0x00FF00 if (trade.pnl or 0) >= 0 else 0xFF6B6B,
            fields=[
                {"name": "Entry", "value": f"${trade.entry_price:,.2f}", "inline": True},
                {"name": "Exit", "value": f"${trade.exit_price:,.2f}", "inline": True},
                {"name": "Duration", "value": entry._format_duration(trade.duration_seconds or 0), "inline": True},
                {"name": "Strategy", "value": trade.entry_strategy, "inline": True},
                {"name": "Score", "value": f"{trade.score:.0f}/100" if trade.score else "Pending", "inline": True},
                {"name": "Lessons", "value": trade.lessons_learned[0] if trade.lessons_learned else "Analyzing...", "inline": False},
            ],
        )

    async def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get daily trading summary.

        Args:
            date: Date to summarize (default: today)

        Returns:
            Dictionary with daily statistics
        """
        if date is None:
            date = datetime.utcnow()

        date_str = date.strftime("%Y-%m-%d")

        # Get all trades for the day
        trades = await self.trade_memory.get_trades_by_date_range(
            start=datetime.strptime(date_str, "%Y-%m-%d"),
            end=datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1),
        )

        if not trades:
            return {
                "date": date_str,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "best_trade": None,
                "worst_trade": None,
            }

        # Calculate statistics
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        wins = [t for t in closed_trades if (t.pnl or 0) > 0]
        losses = [t for t in closed_trades if (t.pnl or 0) < 0]

        total_pnl = sum(t.pnl or 0 for t in closed_trades)
        avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0

        best_trade = max(closed_trades, key=lambda t: t.pnl or 0) if closed_trades else None
        worst_trade = min(closed_trades, key=lambda t: t.pnl or 0) if closed_trades else None

        return {
            "date": date_str,
            "total_trades": len(trades),
            "closed_trades": len(closed_trades),
            "open_trades": len(trades) - len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "best_trade": {
                "id": best_trade.trade_id[:8],
                "pnl": best_trade.pnl,
                "strategy": best_trade.entry_strategy,
            } if best_trade else None,
            "worst_trade": {
                "id": worst_trade.trade_id[:8],
                "pnl": worst_trade.pnl,
                "strategy": worst_trade.entry_strategy,
            } if worst_trade else None,
            "strategies": self._group_by_strategy(closed_trades),
        }

    async def send_daily_summary(self, date: Optional[datetime] = None):
        """Send daily summary to Discord.

        Args:
            date: Date to summarize (default: today)
        """
        if not self.discord:
            return

        summary = await self.get_daily_summary(date)

        if summary["total_trades"] == 0:
            await self.discord.send_status(
                "Daily Journal Summary",
                f"No trades executed on {summary['date']}",
                color=0x808080,
            )
            return

        # Build strategy breakdown
        strategy_lines = []
        for strategy, stats in summary.get("strategies", {}).items():
            strategy_lines.append(
                f"- **{strategy}:** {stats['wins']}W/{stats['losses']}L (${stats['pnl']:+,.2f})"
            )

        await self.discord.send_custom_embed(
            title=f"Daily Journal: {summary['date']}",
            description=f"**Total P&L:** ${summary['total_pnl']:+,.2f}",
            color=0x00FF00 if summary['total_pnl'] >= 0 else 0xFF6B6B,
            fields=[
                {"name": "Trades", "value": f"{summary['closed_trades']} closed, {summary.get('open_trades', 0)} open", "inline": True},
                {"name": "Win Rate", "value": f"{summary['win_rate']*100:.0f}% ({summary['wins']}W/{summary['losses']}L)", "inline": True},
                {"name": "Avg P&L", "value": f"${summary['avg_pnl']:+,.2f}", "inline": True},
                {"name": "Best Trade", "value": f"${summary['best_trade']['pnl']:+,.2f} ({summary['best_trade']['strategy']})" if summary['best_trade'] else "N/A", "inline": True},
                {"name": "Worst Trade", "value": f"${summary['worst_trade']['pnl']:+,.2f} ({summary['worst_trade']['strategy']})" if summary['worst_trade'] else "N/A", "inline": True},
                {"name": "By Strategy", "value": "\n".join(strategy_lines) if strategy_lines else "N/A", "inline": False},
            ],
        )

    async def export_to_markdown(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """Export trade journal to markdown.

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)

        Returns:
            Markdown string of all journal entries
        """
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        trades = await self.trade_memory.get_trades_by_date_range(
            start=start_date,
            end=end_date,
        )

        if not trades:
            return f"# Trade Journal\n\nNo trades found between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}\n"

        lines = [
            f"# Trade Journal",
            f"",
            f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"**Total Trades:** {len(trades)}",
            f"",
        ]

        # Add overall statistics
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        if closed_trades:
            total_pnl = sum(t.pnl or 0 for t in closed_trades)
            wins = len([t for t in closed_trades if (t.pnl or 0) > 0])
            win_rate = wins / len(closed_trades)

            lines.extend([
                f"## Summary Statistics",
                f"",
                f"- **Total P&L:** ${total_pnl:+,.2f}",
                f"- **Win Rate:** {win_rate*100:.1f}% ({wins}/{len(closed_trades)})",
                f"- **Avg P&L:** ${total_pnl/len(closed_trades):+,.2f}",
                f"",
            ])

        # Add individual entries
        lines.append("## Trade Entries")
        lines.append("")

        for trade in sorted(trades, key=lambda t: t.entry_timestamp, reverse=True):
            entry = TradeJournalEntry(trade)
            lines.append(entry.to_markdown())

        return "\n".join(lines)

    def _group_by_strategy(self, trades: List[TradeRecord]) -> Dict[str, Dict]:
        """Group trades by strategy and calculate stats."""
        strategies = {}

        for trade in trades:
            strategy = trade.entry_strategy
            if strategy not in strategies:
                strategies[strategy] = {
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0,
                    "trades": 0,
                }

            strategies[strategy]["trades"] += 1
            strategies[strategy]["pnl"] += trade.pnl or 0

            if (trade.pnl or 0) > 0:
                strategies[strategy]["wins"] += 1
            elif (trade.pnl or 0) < 0:
                strategies[strategy]["losses"] += 1

        return strategies
