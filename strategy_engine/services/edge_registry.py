"""Edge Registry Service for Statistical Edge Collection System.

Manages all registered edges and their statistics in Redis:
- Edge definitions with entry/exit conditions
- Statistical tracking (win rate, payoff, expectancy)
- Rolling performance windows
- Health monitoring and auto-disable

Redis Key Schema:
- edge:{edge_id}                  → Edge JSON (full edge state)
- edge:index:all                  → Set of all edge_ids
- edge:index:symbol:{symbol}      → Set of edges for symbol
- edge:index:type:{edge_type}     → Set of edges by type
- edge:index:active               → Set of active edge_ids
- edge:trade:{edge_id}:{trade_id} → EdgeTradeAttribution JSON
- edge:trades:{edge_id}           → Sorted set of trade_ids by timestamp
"""

import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from .redis_client import RedisClient
from ..models.edge import (
    Edge,
    EdgeStatus,
    EdgeType,
    EdgeSignal,
    EdgeHealth,
    EdgeHealthStatus,
    EdgeTradeAttribution,
    ConvergenceReport,
)


class EdgeRegistry:
    """Manages all registered edges and their statistics.

    Provides async methods for:
    - Registering new edges
    - Updating edge statistics after trades
    - Querying active edges
    - Health monitoring
    """

    # Redis key prefixes
    EDGE_PREFIX = "edge"
    TRADE_PREFIX = "edge:trade"
    TRADES_INDEX = "edge:trades"

    # TTLs
    EDGE_TTL = None  # Edges never expire
    ATTRIBUTION_TTL = 90 * 24 * 3600  # 90 days

    def __init__(self, redis_client: RedisClient):
        """Initialize Edge Registry.

        Args:
            redis_client: Connected Redis client instance
        """
        self.redis = redis_client
        self._initialized = False
        self._edge_cache: Dict[str, Edge] = {}  # In-memory cache for hot edges

    async def initialize(self) -> bool:
        """Initialize the edge registry.

        Returns:
            True if initialization successful
        """
        if not self.redis.is_connected:
            logger.warning("Redis not connected - edge registry running in degraded mode")
            return False

        self._initialized = True
        logger.info("Edge Registry initialized")
        return True

    def _serialize(self, obj: Any) -> str:
        """Serialize object to JSON string."""
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), default=str)
        return json.dumps(obj, default=str)

    def _deserialize(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Deserialize JSON bytes to dict."""
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    # =========================================================================
    # EDGE REGISTRATION
    # =========================================================================

    async def register_edge(self, edge: Edge) -> str:
        """Register a new edge definition.

        Args:
            edge: Edge object to register

        Returns:
            edge_id of the registered edge
        """
        if not self.redis.is_connected:
            logger.warning(f"Redis unavailable - edge {edge.edge_id} not persisted")
            self._edge_cache[edge.edge_id] = edge
            return edge.edge_id

        try:
            client = self.redis.client
            edge_key = f"{self.EDGE_PREFIX}:{edge.edge_id}"

            # Check if edge already exists
            exists = await client.exists(edge_key)
            if exists:
                logger.info(f"Edge {edge.edge_id} already registered, updating")

            # Store edge
            await client.set(edge_key, self._serialize(edge))

            # Add to indexes
            await client.sadd(f"{self.EDGE_PREFIX}:index:all", edge.edge_id)

            # Index by symbol
            if edge.symbol != "*":
                await client.sadd(
                    f"{self.EDGE_PREFIX}:index:symbol:{edge.symbol}",
                    edge.edge_id,
                )

            # Index by type
            await client.sadd(
                f"{self.EDGE_PREFIX}:index:type:{edge.edge_type}",
                edge.edge_id,
            )

            # Add to active index if active
            if edge.status == EdgeStatus.ACTIVE:
                await client.sadd(f"{self.EDGE_PREFIX}:index:active", edge.edge_id)
            else:
                # Remove from active if not active
                await client.srem(f"{self.EDGE_PREFIX}:index:active", edge.edge_id)

            # Update cache
            self._edge_cache[edge.edge_id] = edge

            logger.info(f"Edge registered: {edge.edge_id} ({edge.name})")
            return edge.edge_id

        except Exception as e:
            logger.error(f"Failed to register edge: {e}")
            return edge.edge_id

    async def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID.

        Args:
            edge_id: Edge identifier

        Returns:
            Edge object or None if not found
        """
        # Check cache first
        if edge_id in self._edge_cache:
            return self._edge_cache[edge_id]

        if not self.redis.is_connected:
            return None

        try:
            client = self.redis.client
            data = await client.get(f"{self.EDGE_PREFIX}:{edge_id}")

            if data is None:
                return None

            edge_dict = self._deserialize(data)
            edge = Edge(**edge_dict)

            # Update cache
            self._edge_cache[edge_id] = edge

            return edge

        except Exception as e:
            logger.error(f"Failed to get edge {edge_id}: {e}")
            return None

    async def get_all_edges(self) -> List[Edge]:
        """Get all registered edges.

        Returns:
            List of all Edge objects
        """
        if not self.redis.is_connected:
            return list(self._edge_cache.values())

        try:
            client = self.redis.client
            edge_ids = await client.smembers(f"{self.EDGE_PREFIX}:index:all")

            edges = []
            for edge_id in edge_ids:
                if isinstance(edge_id, bytes):
                    edge_id = edge_id.decode("utf-8")
                edge = await self.get_edge(edge_id)
                if edge:
                    edges.append(edge)

            return edges

        except Exception as e:
            logger.error(f"Failed to get all edges: {e}")
            return list(self._edge_cache.values())

    async def get_active_edges(self, symbol: Optional[str] = None) -> List[Edge]:
        """Get all active edges, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active Edge objects
        """
        if not self.redis.is_connected:
            edges = [e for e in self._edge_cache.values() if e.status == EdgeStatus.ACTIVE]
            if symbol:
                edges = [e for e in edges if e.symbol == symbol or e.symbol == "*"]
            return edges

        try:
            client = self.redis.client

            if symbol:
                # Get intersection of active and symbol-specific edges
                active_ids = await client.smembers(f"{self.EDGE_PREFIX}:index:active")
                symbol_ids = await client.smembers(
                    f"{self.EDGE_PREFIX}:index:symbol:{symbol}"
                )
                wildcard_ids = await client.smembers(
                    f"{self.EDGE_PREFIX}:index:symbol:*"
                )

                # Union of symbol-specific and wildcard
                symbol_edges = symbol_ids | wildcard_ids
                # Intersect with active
                edge_ids = active_ids & symbol_edges if symbol_edges else active_ids
            else:
                edge_ids = await client.smembers(f"{self.EDGE_PREFIX}:index:active")

            edges = []
            for edge_id in edge_ids:
                if isinstance(edge_id, bytes):
                    edge_id = edge_id.decode("utf-8")
                edge = await self.get_edge(edge_id)
                if edge and edge.can_trade:
                    edges.append(edge)

            return edges

        except Exception as e:
            logger.error(f"Failed to get active edges: {e}")
            return []

    async def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type.

        Args:
            edge_type: Type of edge to filter by

        Returns:
            List of Edge objects of that type
        """
        if not self.redis.is_connected:
            return [e for e in self._edge_cache.values() if e.edge_type == edge_type]

        try:
            client = self.redis.client
            edge_ids = await client.smembers(
                f"{self.EDGE_PREFIX}:index:type:{edge_type}"
            )

            edges = []
            for edge_id in edge_ids:
                if isinstance(edge_id, bytes):
                    edge_id = edge_id.decode("utf-8")
                edge = await self.get_edge(edge_id)
                if edge:
                    edges.append(edge)

            return edges

        except Exception as e:
            logger.error(f"Failed to get edges by type: {e}")
            return []

    # =========================================================================
    # EDGE STATISTICS UPDATE
    # =========================================================================

    async def update_edge_stats(
        self,
        edge_id: str,
        is_win: bool,
        return_pct: float,
        trade_timestamp: Optional[datetime] = None,
    ) -> Optional[Edge]:
        """Update edge statistics after a trade completes.

        Args:
            edge_id: Edge identifier
            is_win: Whether the trade was profitable
            return_pct: Return percentage (positive for profit, negative for loss)
            trade_timestamp: Timestamp of the trade (default: now)

        Returns:
            Updated Edge object or None if failed
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            logger.error(f"Edge {edge_id} not found for stats update")
            return None

        try:
            trade_ts = trade_timestamp or datetime.now()
            return_pct_abs = abs(return_pct)

            # Update basic counters
            edge.sample_size += 1

            if is_win:
                edge.wins += 1
                edge.total_win_pct += return_pct_abs
                edge.win_streak += 1
                edge.loss_streak = 0
                edge.max_win_streak = max(edge.max_win_streak, edge.win_streak)
                edge.largest_win_pct = max(edge.largest_win_pct, return_pct_abs)
            else:
                edge.losses += 1
                edge.total_loss_pct += return_pct_abs
                edge.loss_streak += 1
                edge.win_streak = 0
                edge.max_loss_streak = max(edge.max_loss_streak, edge.loss_streak)
                edge.largest_loss_pct = max(edge.largest_loss_pct, return_pct_abs)

            # Recalculate averages
            if edge.wins > 0:
                edge.avg_win_pct = edge.total_win_pct / edge.wins
            if edge.losses > 0:
                edge.avg_loss_pct = edge.total_loss_pct / edge.losses

            # Update rolling 30d stats (simplified - ideally use sorted set with timestamps)
            edge.rolling_30d_trades += 1
            if is_win:
                edge.rolling_30d_wins += 1
            edge.rolling_30d_total_return += return_pct

            # Update timestamps
            edge.last_updated = datetime.now()
            edge.last_trade_at = trade_ts

            # Check if edge should transition to ACTIVE
            if edge.status == EdgeStatus.WARMING and edge.is_mature:
                if edge.is_profitable:
                    edge.status = EdgeStatus.ACTIVE
                    logger.info(
                        f"Edge {edge_id} promoted to ACTIVE "
                        f"(sample={edge.sample_size}, expectancy={edge.expectancy:.2%})"
                    )
                else:
                    edge.status = EdgeStatus.DISABLED
                    edge.pause_reason = f"Negative expectancy after {edge.sample_size} trades"
                    logger.warning(
                        f"Edge {edge_id} disabled - negative expectancy: {edge.expectancy:.2%}"
                    )

            # Persist updates
            await self.register_edge(edge)

            return edge

        except Exception as e:
            logger.error(f"Failed to update edge stats for {edge_id}: {e}")
            return None

    async def update_rolling_window(self, edge_id: str, days: int = 30) -> Optional[Edge]:
        """Recalculate rolling window statistics from trade history.

        Args:
            edge_id: Edge identifier
            days: Number of days for rolling window

        Returns:
            Updated Edge object
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            return None

        try:
            # Get trades from rolling window
            cutoff = datetime.now() - timedelta(days=days)
            attributions = await self._get_edge_trades(edge_id, since=cutoff)

            # Recalculate rolling stats
            edge.rolling_30d_trades = len(attributions)
            edge.rolling_30d_wins = sum(1 for a in attributions if a.is_win)
            edge.rolling_30d_total_return = sum(
                a.return_pct for a in attributions if a.return_pct is not None
            )

            # Persist
            await self.register_edge(edge)

            return edge

        except Exception as e:
            logger.error(f"Failed to update rolling window for {edge_id}: {e}")
            return None

    # =========================================================================
    # TRADE ATTRIBUTION
    # =========================================================================

    async def record_trade_attribution(
        self,
        attribution: EdgeTradeAttribution,
    ) -> str:
        """Record attribution of a trade to an edge.

        Args:
            attribution: EdgeTradeAttribution object

        Returns:
            trade_id
        """
        if not self.redis.is_connected:
            logger.warning(f"Redis unavailable - attribution not persisted")
            return attribution.trade_id

        try:
            client = self.redis.client
            attr_key = f"{self.TRADE_PREFIX}:{attribution.edge_id}:{attribution.trade_id}"

            # Store attribution
            await client.set(
                attr_key,
                self._serialize(attribution),
                ex=self.ATTRIBUTION_TTL,
            )

            # Add to edge's trade index
            await client.zadd(
                f"{self.TRADES_INDEX}:{attribution.edge_id}",
                {attribution.trade_id: attribution.timestamp.timestamp()},
            )

            logger.debug(
                f"Trade attribution recorded: {attribution.trade_id} → {attribution.edge_id}"
            )
            return attribution.trade_id

        except Exception as e:
            logger.error(f"Failed to record trade attribution: {e}")
            return attribution.trade_id

    async def update_trade_attribution(
        self,
        edge_id: str,
        trade_id: str,
        actual_pnl: float,
        return_pct: float,
        is_win: bool,
    ) -> Optional[EdgeTradeAttribution]:
        """Update trade attribution with actual results.

        Args:
            edge_id: Edge identifier
            trade_id: Trade identifier
            actual_pnl: Actual P&L in USD
            return_pct: Actual return percentage
            is_win: Whether trade was profitable

        Returns:
            Updated EdgeTradeAttribution
        """
        if not self.redis.is_connected:
            return None

        try:
            client = self.redis.client
            attr_key = f"{self.TRADE_PREFIX}:{edge_id}:{trade_id}"

            data = await client.get(attr_key)
            if not data:
                logger.warning(f"Attribution not found for {edge_id}:{trade_id}")
                return None

            attr_dict = self._deserialize(data)
            attr = EdgeTradeAttribution(**attr_dict)

            # Update with actual results
            attr.actual_pnl = actual_pnl
            attr.return_pct = return_pct
            attr.is_win = is_win

            # Persist
            await client.set(
                attr_key,
                self._serialize(attr),
                ex=self.ATTRIBUTION_TTL,
            )

            return attr

        except Exception as e:
            logger.error(f"Failed to update trade attribution: {e}")
            return None

    async def _get_edge_trades(
        self,
        edge_id: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[EdgeTradeAttribution]:
        """Get trades for an edge.

        Args:
            edge_id: Edge identifier
            since: Only include trades after this time
            limit: Maximum number of trades to return

        Returns:
            List of EdgeTradeAttribution objects
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client
            index_key = f"{self.TRADES_INDEX}:{edge_id}"

            if since:
                # Get trades after timestamp
                min_score = since.timestamp()
                trade_ids = await client.zrangebyscore(
                    index_key, min_score, "+inf", start=0, num=limit
                )
            else:
                # Get most recent trades
                trade_ids = await client.zrevrange(index_key, 0, limit - 1)

            attributions = []
            for trade_id in trade_ids:
                if isinstance(trade_id, bytes):
                    trade_id = trade_id.decode("utf-8")

                attr_key = f"{self.TRADE_PREFIX}:{edge_id}:{trade_id}"
                data = await client.get(attr_key)

                if data:
                    attr_dict = self._deserialize(data)
                    attributions.append(EdgeTradeAttribution(**attr_dict))

            return attributions

        except Exception as e:
            logger.error(f"Failed to get edge trades: {e}")
            return []

    # =========================================================================
    # EDGE HEALTH MONITORING
    # =========================================================================

    async def check_edge_health(self, edge_id: str) -> Optional[EdgeHealth]:
        """Check health of an edge by comparing rolling vs historical performance.

        Args:
            edge_id: Edge identifier

        Returns:
            EdgeHealth assessment
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            return None

        # Calculate health metrics
        historical_expectancy = edge.expectancy
        rolling_expectancy = edge.rolling_30d_expectancy

        # Determine status
        if rolling_expectancy < 0:
            status = EdgeHealthStatus.BROKEN
            status_reason = f"Negative rolling expectancy: {rolling_expectancy:.2%}"
            should_pause = True
            should_disable = edge.rolling_30d_trades >= 10  # Enough data to conclude
        elif historical_expectancy > 0 and rolling_expectancy < historical_expectancy * 0.5:
            status = EdgeHealthStatus.DEGRADING
            status_reason = f"Rolling at {rolling_expectancy/historical_expectancy:.0%} of historical"
            should_pause = rolling_expectancy < historical_expectancy * 0.3
            should_disable = False
        else:
            status = EdgeHealthStatus.HEALTHY
            status_reason = "Performing as expected"
            should_pause = False
            should_disable = False

        # Calculate expectancy ratio
        expectancy_ratio = 0.0
        if historical_expectancy > 0:
            expectancy_ratio = rolling_expectancy / historical_expectancy

        return EdgeHealth(
            edge_id=edge_id,
            status=status,
            status_reason=status_reason,
            historical_expectancy=historical_expectancy,
            rolling_expectancy=rolling_expectancy,
            expectancy_ratio=expectancy_ratio,
            historical_win_rate=edge.win_rate,
            rolling_win_rate=edge.rolling_30d_win_rate,
            total_trades=edge.sample_size,
            rolling_trades=edge.rolling_30d_trades,
            should_pause=should_pause,
            should_disable=should_disable,
        )

    async def pause_edge(self, edge_id: str, reason: str) -> bool:
        """Pause an edge (temporarily disable trading).

        Args:
            edge_id: Edge identifier
            reason: Reason for pausing

        Returns:
            True if paused successfully
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            return False

        edge.status = EdgeStatus.PAUSED
        edge.pause_reason = reason
        edge.last_updated = datetime.now()

        await self.register_edge(edge)

        logger.warning(f"Edge {edge_id} paused: {reason}")
        return True

    async def unpause_edge(self, edge_id: str) -> bool:
        """Unpause an edge (re-enable trading).

        Args:
            edge_id: Edge identifier

        Returns:
            True if unpaused successfully
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            return False

        if edge.status != EdgeStatus.PAUSED:
            logger.warning(f"Edge {edge_id} is not paused")
            return False

        edge.status = EdgeStatus.ACTIVE
        edge.pause_reason = None
        edge.last_updated = datetime.now()

        await self.register_edge(edge)

        logger.info(f"Edge {edge_id} unpaused")
        return True

    async def disable_edge(self, edge_id: str, reason: str) -> bool:
        """Permanently disable an edge.

        Args:
            edge_id: Edge identifier
            reason: Reason for disabling

        Returns:
            True if disabled successfully
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            return False

        edge.status = EdgeStatus.DISABLED
        edge.pause_reason = reason
        edge.last_updated = datetime.now()

        await self.register_edge(edge)

        logger.warning(f"Edge {edge_id} permanently disabled: {reason}")
        return True

    # =========================================================================
    # CONVERGENCE ANALYSIS
    # =========================================================================

    async def get_convergence_report(
        self,
        edge_id: str,
        period_days: int = 30,
    ) -> Optional[ConvergenceReport]:
        """Check if actual results are converging to expected edge.

        Args:
            edge_id: Edge identifier
            period_days: Analysis period in days

        Returns:
            ConvergenceReport
        """
        edge = await self.get_edge(edge_id)
        if not edge:
            return None

        # Get attributions from period
        cutoff = datetime.now() - timedelta(days=period_days)
        attributions = await self._get_edge_trades(edge_id, since=cutoff)

        if not attributions:
            return ConvergenceReport(
                edge_id=edge_id,
                sample_size=0,
                period_days=period_days,
                assessment="No trades in period",
            )

        # Calculate expected vs actual
        expected_pnl = sum(a.expected_pnl for a in attributions)
        actual_pnl = sum(
            a.actual_pnl for a in attributions if a.actual_pnl is not None
        )

        # Convergence ratio
        convergence_ratio = 0.0
        if expected_pnl != 0:
            convergence_ratio = actual_pnl / expected_pnl

        # Determine status
        if convergence_ratio > 1.3:
            convergence_status = "OUTPERFORMING"
        elif convergence_ratio < 0.7:
            convergence_status = "UNDERPERFORMING"
        else:
            convergence_status = "CONVERGING"

        # Expected variance (rough approximation)
        # Var[sum] = n * variance_per_trade
        # For binomial-ish outcomes with returns
        n = len(attributions)
        variance_per_trade = (edge.avg_win_pct ** 2 * edge.win_rate +
                             edge.avg_loss_pct ** 2 * edge.loss_rate)
        expected_variance = n * variance_per_trade
        expected_std = math.sqrt(expected_variance) if expected_variance > 0 else 0

        # Check if within 2 std
        within_2_std = abs(actual_pnl - expected_pnl) <= 2 * expected_std

        # Assessment
        if convergence_status == "CONVERGING":
            assessment = f"Edge is performing as expected (ratio: {convergence_ratio:.1%})"
        elif convergence_status == "OUTPERFORMING":
            if within_2_std:
                assessment = "Edge outperforming but within statistical bounds - likely variance"
            else:
                assessment = "Edge significantly outperforming - may have improved or got lucky"
        else:
            if within_2_std:
                assessment = "Edge underperforming but within statistical bounds - likely variance"
            else:
                assessment = "Edge significantly underperforming - may have degraded"

        return ConvergenceReport(
            edge_id=edge_id,
            sample_size=len(attributions),
            period_days=period_days,
            expected_pnl=expected_pnl,
            actual_pnl=actual_pnl,
            convergence_ratio=convergence_ratio,
            is_converging=convergence_status == "CONVERGING",
            convergence_status=convergence_status,
            expected_variance=expected_variance,
            within_2_std=within_2_std,
            assessment=assessment,
        )

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    async def refresh_cache(self):
        """Refresh the in-memory cache from Redis."""
        self._edge_cache.clear()
        await self.get_all_edges()  # This populates cache
        logger.info(f"Edge cache refreshed: {len(self._edge_cache)} edges")

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._edge_cache.clear()
        logger.info("Edge cache cleared")

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================

    async def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the edge registry.

        Returns:
            Summary dictionary
        """
        all_edges = await self.get_all_edges()
        active_edges = [e for e in all_edges if e.status == EdgeStatus.ACTIVE]
        warming_edges = [e for e in all_edges if e.status == EdgeStatus.WARMING]
        paused_edges = [e for e in all_edges if e.status == EdgeStatus.PAUSED]
        disabled_edges = [e for e in all_edges if e.status == EdgeStatus.DISABLED]

        # Calculate aggregate stats
        total_trades = sum(e.sample_size for e in all_edges)
        total_wins = sum(e.wins for e in all_edges)
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0

        # Average expectancy of active edges
        active_expectancies = [e.expectancy for e in active_edges if e.expectancy > 0]
        avg_expectancy = (
            sum(active_expectancies) / len(active_expectancies)
            if active_expectancies else 0
        )

        return {
            "total_edges": len(all_edges),
            "active_edges": len(active_edges),
            "warming_edges": len(warming_edges),
            "paused_edges": len(paused_edges),
            "disabled_edges": len(disabled_edges),
            "total_trades": total_trades,
            "overall_win_rate": overall_win_rate,
            "avg_active_expectancy": avg_expectancy,
            "by_type": {
                edge_type.value: len([e for e in all_edges if e.edge_type == edge_type])
                for edge_type in EdgeType
            },
        }
