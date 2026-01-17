"""Performance Tracker Service for Statistical Edge Collection System.

Tracks performance and convergence to expected edge value:
- Records trade results and attributes them to edges
- Compares actual P&L vs expected P&L
- Monitors convergence over time
- Alerts when edges are underperforming

Key metrics:
- Expected P&L: edge.expectancy * position_size * entry_price
- Actual P&L: realized P&L from the trade
- Convergence ratio: actual_pnl / expected_pnl
- Within 2 std: statistical significance check
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from .edge_registry import EdgeRegistry
from .redis_client import RedisClient
from ..models.edge import (
    Edge,
    EdgeTradeAttribution,
    ConvergenceReport,
    EdgeHealth,
    EdgeHealthStatus,
)


class PerformanceTracker:
    """Tracks performance and convergence to expected edge.

    Records trade attributions, updates edge statistics, and monitors
    whether actual results are converging to expected values.
    """

    def __init__(
        self,
        edge_registry: EdgeRegistry,
        redis_client: Optional[RedisClient] = None,
        discord=None,
    ):
        """Initialize Performance Tracker.

        Args:
            edge_registry: EdgeRegistry for updating edge stats
            redis_client: Optional Redis client for additional storage
            discord: Optional Discord notifier for alerts
        """
        self.edge_registry = edge_registry
        self.redis = redis_client
        self.discord = discord
        self._attribution_cache: Dict[str, EdgeTradeAttribution] = {}

    async def record_entry(
        self,
        edge_id: str,
        trade_id: str,
        position_size: float,
        entry_price: float,
        kelly_used: float,
    ) -> Optional[EdgeTradeAttribution]:
        """Record a trade entry for an edge.

        Args:
            edge_id: Edge identifier
            trade_id: Trade identifier
            position_size: Position size in base units
            entry_price: Entry price
            kelly_used: Kelly fraction used for sizing

        Returns:
            EdgeTradeAttribution object
        """
        edge = await self.edge_registry.get_edge(edge_id)
        if not edge:
            logger.error(f"Edge {edge_id} not found for entry attribution")
            return None

        # Calculate expected P&L
        position_value = position_size * entry_price
        expected_pnl = edge.expectancy * position_value

        attribution = EdgeTradeAttribution(
            trade_id=trade_id,
            edge_id=edge_id,
            expected_pnl=expected_pnl,
            edge_expectancy_at_entry=edge.expectancy,
            edge_win_rate_at_entry=edge.win_rate,
            edge_sample_size_at_entry=edge.sample_size,
            position_size=position_size,
            kelly_used=kelly_used,
        )

        # Store in registry
        await self.edge_registry.record_trade_attribution(attribution)

        # Cache for quick lookup
        self._attribution_cache[trade_id] = attribution

        logger.info(
            f"Trade {trade_id} attributed to edge {edge_id} "
            f"(expected P&L: ${expected_pnl:.2f})"
        )

        return attribution

    async def record_exit(
        self,
        trade_id: str,
        actual_pnl: float,
        return_pct: float,
    ) -> Optional[EdgeTradeAttribution]:
        """Record a trade exit and update edge statistics.

        Args:
            trade_id: Trade identifier
            actual_pnl: Actual P&L in USD
            return_pct: Return percentage

        Returns:
            Updated EdgeTradeAttribution
        """
        # Get attribution from cache or registry
        attribution = self._attribution_cache.get(trade_id)

        if not attribution:
            logger.warning(f"No attribution found for trade {trade_id}")
            return None

        is_win = return_pct > 0
        edge_id = attribution.edge_id

        # Update attribution
        updated = await self.edge_registry.update_trade_attribution(
            edge_id=edge_id,
            trade_id=trade_id,
            actual_pnl=actual_pnl,
            return_pct=return_pct,
            is_win=is_win,
        )

        # Update edge statistics
        await self.edge_registry.update_edge_stats(
            edge_id=edge_id,
            is_win=is_win,
            return_pct=return_pct,
        )

        # Check convergence
        convergence = await self.edge_registry.get_convergence_report(edge_id)
        if convergence:
            await self._check_convergence_alert(edge_id, convergence)

        # Remove from cache
        if trade_id in self._attribution_cache:
            del self._attribution_cache[trade_id]

        logger.info(
            f"Trade {trade_id} exit recorded: {'+' if is_win else ''}{return_pct:.2f}% "
            f"(expected: ${attribution.expected_pnl:.2f}, actual: ${actual_pnl:.2f})"
        )

        return updated

    async def _check_convergence_alert(
        self,
        edge_id: str,
        convergence: ConvergenceReport,
    ):
        """Check if convergence warrants an alert.

        Args:
            edge_id: Edge identifier
            convergence: Convergence report
        """
        if convergence.convergence_status == "UNDERPERFORMING":
            if not convergence.within_2_std:
                # Significant underperformance - alert
                logger.warning(
                    f"Edge {edge_id} significantly underperforming: "
                    f"expected ${convergence.expected_pnl:.2f}, "
                    f"actual ${convergence.actual_pnl:.2f} "
                    f"(ratio: {convergence.convergence_ratio:.0%})"
                )

                if self.discord:
                    await self.discord.send_alert(
                        title=f"Edge Underperforming: {edge_id}",
                        description=convergence.assessment,
                        color=0xFFA500,  # Orange
                        fields={
                            "Expected P&L": f"${convergence.expected_pnl:.2f}",
                            "Actual P&L": f"${convergence.actual_pnl:.2f}",
                            "Convergence Ratio": f"{convergence.convergence_ratio:.0%}",
                            "Sample Size": str(convergence.sample_size),
                        },
                    )

    # =========================================================================
    # EDGE HEALTH MONITORING
    # =========================================================================

    async def check_all_edge_health(self) -> List[EdgeHealth]:
        """Check health of all active edges.

        Returns:
            List of EdgeHealth objects with status
        """
        active_edges = await self.edge_registry.get_active_edges()
        health_reports = []

        for edge in active_edges:
            health = await self.edge_registry.check_edge_health(edge.edge_id)
            if health:
                health_reports.append(health)

                # Take action based on health
                if health.should_disable:
                    await self._handle_broken_edge(edge.edge_id, health)
                elif health.should_pause:
                    await self._handle_degrading_edge(edge.edge_id, health)

        return health_reports

    async def _handle_broken_edge(self, edge_id: str, health: EdgeHealth):
        """Handle a broken edge (negative expectancy).

        Args:
            edge_id: Edge identifier
            health: EdgeHealth assessment
        """
        logger.error(
            f"Edge {edge_id} has broken - disabling: {health.status_reason}"
        )

        # Disable the edge
        await self.edge_registry.disable_edge(
            edge_id,
            f"Broken: {health.status_reason}",
        )

        # Notify
        if self.discord:
            await self.discord.send_alert(
                title=f"Edge Disabled: {edge_id}",
                description=f"Edge has been automatically disabled due to negative expectancy.",
                color=0xFF0000,  # Red
                fields={
                    "Status": health.status.value,
                    "Reason": health.status_reason,
                    "Historical Expectancy": f"{health.historical_expectancy:.2%}",
                    "Rolling Expectancy": f"{health.rolling_expectancy:.2%}",
                    "Rolling Trades": str(health.rolling_trades),
                },
            )

    async def _handle_degrading_edge(self, edge_id: str, health: EdgeHealth):
        """Handle a degrading edge (significantly underperforming).

        Args:
            edge_id: Edge identifier
            health: EdgeHealth assessment
        """
        logger.warning(
            f"Edge {edge_id} is degrading - pausing: {health.status_reason}"
        )

        # Pause the edge
        await self.edge_registry.pause_edge(
            edge_id,
            f"Degrading: {health.status_reason}",
        )

        # Notify
        if self.discord:
            await self.discord.send_alert(
                title=f"Edge Paused: {edge_id}",
                description=f"Edge has been automatically paused due to degrading performance.",
                color=0xFFA500,  # Orange
                fields={
                    "Status": health.status.value,
                    "Reason": health.status_reason,
                    "Historical Expectancy": f"{health.historical_expectancy:.2%}",
                    "Rolling Expectancy": f"{health.rolling_expectancy:.2%}",
                    "Expectancy Ratio": f"{health.expectancy_ratio:.0%}",
                },
            )

    # =========================================================================
    # REPORTING
    # =========================================================================

    async def get_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report.

        Returns:
            Report dictionary
        """
        all_edges = await self.edge_registry.get_all_edges()
        registry_summary = await self.edge_registry.get_registry_summary()

        # Get convergence for each active edge
        convergence_reports = []
        for edge in all_edges:
            if edge.sample_size > 0:
                convergence = await self.edge_registry.get_convergence_report(edge.edge_id)
                if convergence:
                    convergence_reports.append({
                        "edge_id": edge.edge_id,
                        "name": edge.name,
                        "sample_size": convergence.sample_size,
                        "expected_pnl": convergence.expected_pnl,
                        "actual_pnl": convergence.actual_pnl,
                        "convergence_ratio": convergence.convergence_ratio,
                        "status": convergence.convergence_status,
                    })

        # Best and worst performing edges
        sorted_edges = sorted(all_edges, key=lambda e: e.expectancy, reverse=True)
        best_edges = [
            {"edge_id": e.edge_id, "name": e.name, "expectancy": e.expectancy}
            for e in sorted_edges[:3] if e.expectancy > 0
        ]
        worst_edges = [
            {"edge_id": e.edge_id, "name": e.name, "expectancy": e.expectancy}
            for e in sorted_edges[-3:] if e.expectancy < 0
        ]

        return {
            "timestamp": datetime.now().isoformat(),
            "registry_summary": registry_summary,
            "convergence_reports": convergence_reports,
            "best_edges": best_edges,
            "worst_edges": worst_edges,
            "recommendations": await self._generate_recommendations(all_edges),
        }

    async def _generate_recommendations(self, edges: List[Edge]) -> List[str]:
        """Generate recommendations based on edge performance.

        Args:
            edges: List of all edges

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for warming edges ready to activate
        warming_ready = [
            e for e in edges
            if e.status.value == "warming" and e.is_mature and e.is_profitable
        ]
        if warming_ready:
            recommendations.append(
                f"{len(warming_ready)} edge(s) ready to activate: "
                f"{', '.join(e.edge_id for e in warming_ready)}"
            )

        # Check for paused edges that could be unpaused
        paused = [e for e in edges if e.status.value == "paused"]
        for edge in paused:
            if edge.rolling_30d_expectancy > edge.min_expectancy:
                recommendations.append(
                    f"Edge {edge.edge_id} may be ready to unpause "
                    f"(rolling expectancy: {edge.rolling_30d_expectancy:.2%})"
                )

        # Check for edges with low sample sizes
        low_sample = [
            e for e in edges
            if e.status.value == "active" and e.sample_size < e.min_sample_size * 1.5
        ]
        if low_sample:
            recommendations.append(
                f"{len(low_sample)} active edge(s) have low sample sizes - "
                "statistics may be unreliable"
            )

        # Check for high loss streaks
        high_loss_streak = [e for e in edges if e.loss_streak >= 5]
        for edge in high_loss_streak:
            recommendations.append(
                f"Edge {edge.edge_id} on {edge.loss_streak}-trade losing streak - "
                "consider review"
            )

        return recommendations

    async def get_edge_performance_summary(self, edge_id: str) -> Dict[str, Any]:
        """Get detailed performance summary for an edge.

        Args:
            edge_id: Edge identifier

        Returns:
            Performance summary dictionary
        """
        edge = await self.edge_registry.get_edge(edge_id)
        if not edge:
            return {"error": f"Edge {edge_id} not found"}

        convergence = await self.edge_registry.get_convergence_report(edge_id)
        health = await self.edge_registry.check_edge_health(edge_id)

        return {
            "edge_id": edge_id,
            "name": edge.name,
            "edge_type": edge.edge_type,
            "symbol": edge.symbol,
            "timeframe": edge.timeframe,
            "status": edge.status,
            "statistics": {
                "sample_size": edge.sample_size,
                "win_rate": edge.win_rate,
                "avg_win_pct": edge.avg_win_pct,
                "avg_loss_pct": edge.avg_loss_pct,
                "payoff_ratio": edge.payoff_ratio,
                "expectancy": edge.expectancy,
                "kelly_fraction": edge.kelly_fraction,
            },
            "streaks": {
                "current_win_streak": edge.win_streak,
                "current_loss_streak": edge.loss_streak,
                "max_win_streak": edge.max_win_streak,
                "max_loss_streak": edge.max_loss_streak,
            },
            "rolling_30d": {
                "trades": edge.rolling_30d_trades,
                "win_rate": edge.rolling_30d_win_rate,
                "expectancy": edge.rolling_30d_expectancy,
            },
            "convergence": {
                "expected_pnl": convergence.expected_pnl if convergence else 0,
                "actual_pnl": convergence.actual_pnl if convergence else 0,
                "ratio": convergence.convergence_ratio if convergence else 0,
                "status": convergence.convergence_status if convergence else "N/A",
                "within_2_std": convergence.within_2_std if convergence else True,
            } if convergence else None,
            "health": {
                "status": health.status.value if health else "unknown",
                "reason": health.status_reason if health else "",
                "should_pause": health.should_pause if health else False,
                "should_disable": health.should_disable if health else False,
            } if health else None,
        }
