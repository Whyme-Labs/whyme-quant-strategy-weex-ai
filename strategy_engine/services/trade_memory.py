"""Trade Memory Service for Triple Memory System.

Manages three types of memory in Redis:
1. Episodic Memory: Individual trade records with full context
2. Semantic Memory: Extracted patterns and insights
3. Procedural Memory: Strategy parameters and their evolution

Redis Key Schema:
- memory:episodic:{trade_id}           → TradeRecord JSON
- memory:episodic:index:{symbol}       → Sorted set by timestamp
- memory:episodic:index:strategy:{s}   → Sorted set by timestamp
- memory:episodic:open                 → Set of open trade_ids
- memory:semantic:{insight_id}         → StrategyInsight JSON
- memory:semantic:index                → Set of all insight_ids
- memory:procedural:{agent}:{param}    → ParameterState JSON
- memory:procedural:index:{agent}      → Set of parameter names
- memory:procedural:evolution          → List of ParameterEvolution JSONs
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from loguru import logger

from .redis_client import RedisClient
from ..models.memory import (
    TradeRecord,
    TradeStatus,
    ExitReason,
    StrategyInsight,
    ParameterState,
    ParameterEvolution,
    TradeScore,
    DailyLearningReport,
)


class TradeMemoryService:
    """Manages Triple Memory System in Redis.

    Provides async methods for storing and retrieving:
    - Trade records (Episodic Memory)
    - Strategy insights (Semantic Memory)
    - Parameter states (Procedural Memory)
    """

    # Redis key prefixes
    EPISODIC_PREFIX = "memory:episodic"
    SEMANTIC_PREFIX = "memory:semantic"
    PROCEDURAL_PREFIX = "memory:procedural"
    METRICS_PREFIX = "metrics"

    # TTLs (in seconds)
    TRADE_TTL = 90 * 24 * 3600  # 90 days for trade records
    INSIGHT_TTL = 365 * 24 * 3600  # 1 year for insights
    PARAM_TTL = None  # Parameters never expire

    def __init__(self, redis_client: RedisClient):
        """Initialize Trade Memory Service.

        Args:
            redis_client: Connected Redis client instance
        """
        self.redis = redis_client
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the memory service.

        Returns:
            True if initialization successful
        """
        if not self.redis.is_connected:
            logger.warning("Redis not connected - memory service running in degraded mode")
            return False

        self._initialized = True
        logger.info("Trade Memory Service initialized")
        return True

    def _serialize(self, obj: Any) -> str:
        """Serialize object to JSON string."""
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), default=str)
        return json.dumps(obj, default=str)

    def _deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize JSON bytes to dict."""
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    # =========================================================================
    # EPISODIC MEMORY (Trade Records)
    # =========================================================================

    async def record_trade_entry(self, trade: TradeRecord) -> str:
        """Record a new trade entry in episodic memory.

        Args:
            trade: TradeRecord with entry information

        Returns:
            trade_id of the recorded trade
        """
        if not self.redis.is_connected:
            logger.warning(f"Redis unavailable - trade {trade.trade_id} not persisted")
            return trade.trade_id

        try:
            client = self.redis.client
            trade_key = f"{self.EPISODIC_PREFIX}:{trade.trade_id}"

            # Store trade record
            await client.set(
                trade_key,
                self._serialize(trade),
                ex=self.TRADE_TTL,
            )

            # Add to indexes
            timestamp = trade.entry_timestamp.timestamp()

            # Index by symbol
            await client.zadd(
                f"{self.EPISODIC_PREFIX}:index:{trade.symbol}",
                {trade.trade_id: timestamp},
            )

            # Index by strategy
            await client.zadd(
                f"{self.EPISODIC_PREFIX}:index:strategy:{trade.entry_strategy}",
                {trade.trade_id: timestamp},
            )

            # Add to open trades set
            if trade.status == TradeStatus.OPEN:
                await client.sadd(f"{self.EPISODIC_PREFIX}:open", trade.trade_id)

            logger.info(f"Trade entry recorded: {trade.trade_id} ({trade.symbol})")
            return trade.trade_id

        except Exception as e:
            logger.error(f"Failed to record trade entry: {e}")
            return trade.trade_id

    async def record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: ExitReason,
        exit_regime: Optional[Dict[str, str]] = None,
    ) -> Optional[TradeRecord]:
        """Record trade exit and calculate outcomes.

        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price
            exit_reason: Reason for exit
            exit_regime: Market regime at exit

        Returns:
            Updated TradeRecord or None if not found
        """
        if not self.redis.is_connected:
            logger.warning(f"Redis unavailable - cannot record exit for {trade_id}")
            return None

        try:
            # Get existing trade
            trade = await self.get_trade(trade_id)
            if not trade:
                logger.error(f"Trade not found: {trade_id}")
                return None

            # Calculate outcomes
            exit_timestamp = datetime.now()
            duration = int((exit_timestamp - trade.entry_timestamp).total_seconds())

            # Calculate P&L
            if trade.entry_side == "long":
                pnl = (exit_price - trade.entry_price) * trade.entry_size
                pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
            else:  # short
                pnl = (trade.entry_price - exit_price) * trade.entry_size
                pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

            # Update trade record
            trade.exit_timestamp = exit_timestamp
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.exit_regime = exit_regime or {}
            trade.pnl = pnl
            trade.pnl_pct = pnl_pct
            trade.duration_seconds = duration
            trade.status = TradeStatus.CLOSED

            # Store updated record
            client = self.redis.client
            trade_key = f"{self.EPISODIC_PREFIX}:{trade_id}"
            await client.set(trade_key, self._serialize(trade), ex=self.TRADE_TTL)

            # Remove from open trades
            await client.srem(f"{self.EPISODIC_PREFIX}:open", trade_id)

            logger.info(
                f"Trade exit recorded: {trade_id} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | "
                f"Reason: {exit_reason.value}"
            )
            return trade

        except Exception as e:
            logger.error(f"Failed to record trade exit: {e}")
            return None

    async def add_reflection(
        self,
        trade_id: str,
        reflection: str,
        score: float,
        lessons: List[str],
        score_breakdown: Optional[Dict[str, float]] = None,
    ) -> Optional[TradeRecord]:
        """Add LLM reflection and score to a closed trade.

        Args:
            trade_id: Trade to update
            reflection: LLM-generated reflection
            score: Judge agent score (0-100)
            lessons: List of lessons learned
            score_breakdown: Score by category

        Returns:
            Updated TradeRecord or None
        """
        if not self.redis.is_connected:
            return None

        try:
            trade = await self.get_trade(trade_id)
            if not trade:
                return None

            trade.reflection = reflection
            trade.score = score
            trade.lessons_learned = lessons
            trade.score_breakdown = score_breakdown

            # Store updated record
            client = self.redis.client
            trade_key = f"{self.EPISODIC_PREFIX}:{trade_id}"
            await client.set(trade_key, self._serialize(trade), ex=self.TRADE_TTL)

            logger.info(f"Reflection added to trade {trade_id}: score={score:.1f}")
            return trade

        except Exception as e:
            logger.error(f"Failed to add reflection: {e}")
            return None

    async def update_excursions(
        self,
        trade_id: str,
        mfe: float,
        mae: float,
    ) -> Optional[TradeRecord]:
        """Update max favorable/adverse excursion for a trade.

        Args:
            trade_id: Trade to update
            mfe: Max favorable excursion (best unrealized P&L)
            mae: Max adverse excursion (worst unrealized P&L)

        Returns:
            Updated TradeRecord or None
        """
        if not self.redis.is_connected:
            return None

        try:
            trade = await self.get_trade(trade_id)
            if not trade:
                return None

            # Update only if new values are more extreme
            if trade.max_favorable_excursion is None or mfe > trade.max_favorable_excursion:
                trade.max_favorable_excursion = mfe
            if trade.max_adverse_excursion is None or mae < trade.max_adverse_excursion:
                trade.max_adverse_excursion = mae

            client = self.redis.client
            trade_key = f"{self.EPISODIC_PREFIX}:{trade_id}"
            await client.set(trade_key, self._serialize(trade), ex=self.TRADE_TTL)
            return trade

        except Exception as e:
            logger.error(f"Failed to update excursions: {e}")
            return None

    async def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a trade record by ID.

        Args:
            trade_id: Trade identifier

        Returns:
            TradeRecord or None if not found
        """
        if not self.redis.is_connected:
            return None

        try:
            client = self.redis.client
            trade_key = f"{self.EPISODIC_PREFIX}:{trade_id}"
            data = await client.get(trade_key)
            if data:
                return TradeRecord(**self._deserialize(data))
            return None

        except Exception as e:
            logger.error(f"Failed to get trade: {e}")
            return None

    async def get_open_trades(self) -> List[TradeRecord]:
        """Get all open trades.

        Returns:
            List of open TradeRecords
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client
            trade_ids = await client.smembers(f"{self.EPISODIC_PREFIX}:open")

            trades = []
            for trade_id in trade_ids:
                if isinstance(trade_id, bytes):
                    trade_id = trade_id.decode("utf-8")
                trade = await self.get_trade(trade_id)
                if trade:
                    trades.append(trade)
            return trades

        except Exception as e:
            logger.error(f"Failed to get open trades: {e}")
            return []

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 50,
        include_open: bool = True,
    ) -> List[TradeRecord]:
        """Get recent trades for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            include_open: Include open trades

        Returns:
            List of TradeRecords sorted by entry time (most recent first)
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client
            index_key = f"{self.EPISODIC_PREFIX}:index:{symbol}"

            # Get trade IDs sorted by timestamp (descending)
            trade_ids = await client.zrevrange(index_key, 0, limit - 1)

            trades = []
            for trade_id in trade_ids:
                if isinstance(trade_id, bytes):
                    trade_id = trade_id.decode("utf-8")
                trade = await self.get_trade(trade_id)
                if trade:
                    if include_open or trade.status == TradeStatus.CLOSED:
                        trades.append(trade)

            return trades

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    async def get_trades_by_strategy(
        self,
        strategy: str,
        limit: int = 100,
    ) -> List[TradeRecord]:
        """Get trades by strategy.

        Args:
            strategy: Strategy name
            limit: Maximum number of trades

        Returns:
            List of TradeRecords
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client
            index_key = f"{self.EPISODIC_PREFIX}:index:strategy:{strategy}"

            trade_ids = await client.zrevrange(index_key, 0, limit - 1)

            trades = []
            for trade_id in trade_ids:
                if isinstance(trade_id, bytes):
                    trade_id = trade_id.decode("utf-8")
                trade = await self.get_trade(trade_id)
                if trade:
                    trades.append(trade)

            return trades

        except Exception as e:
            logger.error(f"Failed to get trades by strategy: {e}")
            return []

    async def get_trades_since(
        self,
        since: datetime,
        symbol: Optional[str] = None,
    ) -> List[TradeRecord]:
        """Get all trades since a given timestamp.

        Args:
            since: Starting timestamp
            symbol: Optional symbol filter

        Returns:
            List of TradeRecords
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client

            if symbol:
                index_key = f"{self.EPISODIC_PREFIX}:index:{symbol}"
            else:
                # Get from all symbols - this is less efficient
                # but we need to scan all indices
                all_keys = await client.keys(f"{self.EPISODIC_PREFIX}:index:*")
                trades = []
                for key in all_keys:
                    if b"strategy" not in key:  # Skip strategy indices
                        trade_ids = await client.zrangebyscore(
                            key, since.timestamp(), "+inf"
                        )
                        for trade_id in trade_ids:
                            if isinstance(trade_id, bytes):
                                trade_id = trade_id.decode("utf-8")
                            trade = await self.get_trade(trade_id)
                            if trade:
                                trades.append(trade)
                return sorted(trades, key=lambda t: t.entry_timestamp, reverse=True)

            trade_ids = await client.zrangebyscore(
                index_key, since.timestamp(), "+inf"
            )

            trades = []
            for trade_id in trade_ids:
                if isinstance(trade_id, bytes):
                    trade_id = trade_id.decode("utf-8")
                trade = await self.get_trade(trade_id)
                if trade:
                    trades.append(trade)

            return sorted(trades, key=lambda t: t.entry_timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Failed to get trades since {since}: {e}")
            return []

    # =========================================================================
    # SEMANTIC MEMORY (Strategy Insights)
    # =========================================================================

    async def store_insight(self, insight: StrategyInsight) -> str:
        """Store a strategy insight in semantic memory.

        Args:
            insight: StrategyInsight to store

        Returns:
            insight_id
        """
        if not self.redis.is_connected:
            logger.warning(f"Redis unavailable - insight {insight.insight_id} not persisted")
            return insight.insight_id

        try:
            client = self.redis.client
            insight_key = f"{self.SEMANTIC_PREFIX}:{insight.insight_id}"

            # Store insight
            await client.set(
                insight_key,
                self._serialize(insight),
                ex=self.INSIGHT_TTL,
            )

            # Add to index
            await client.sadd(f"{self.SEMANTIC_PREFIX}:index", insight.insight_id)

            # Index by pattern
            await client.sadd(
                f"{self.SEMANTIC_PREFIX}:pattern:{insight.pattern}",
                insight.insight_id,
            )

            logger.info(f"Insight stored: {insight.insight_id} ({insight.pattern})")
            return insight.insight_id

        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
            return insight.insight_id

    async def get_insight(self, insight_id: str) -> Optional[StrategyInsight]:
        """Get an insight by ID.

        Args:
            insight_id: Insight identifier

        Returns:
            StrategyInsight or None
        """
        if not self.redis.is_connected:
            return None

        try:
            client = self.redis.client
            insight_key = f"{self.SEMANTIC_PREFIX}:{insight_id}"
            data = await client.get(insight_key)
            if data:
                return StrategyInsight(**self._deserialize(data))
            return None

        except Exception as e:
            logger.error(f"Failed to get insight: {e}")
            return None

    async def get_all_insights(self) -> List[StrategyInsight]:
        """Get all stored insights.

        Returns:
            List of StrategyInsights
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client
            insight_ids = await client.smembers(f"{self.SEMANTIC_PREFIX}:index")

            insights = []
            for insight_id in insight_ids:
                if isinstance(insight_id, bytes):
                    insight_id = insight_id.decode("utf-8")
                insight = await self.get_insight(insight_id)
                if insight:
                    insights.append(insight)

            return sorted(insights, key=lambda i: i.confidence, reverse=True)

        except Exception as e:
            logger.error(f"Failed to get all insights: {e}")
            return []

    async def get_insights_for_conditions(
        self,
        regime: Dict[str, str],
        strategy: str,
    ) -> List[StrategyInsight]:
        """Get insights that apply to given conditions.

        Args:
            regime: Current market regime
            strategy: Strategy being considered

        Returns:
            List of applicable StrategyInsights
        """
        all_insights = await self.get_all_insights()

        applicable = []
        for insight in all_insights:
            applies = insight.applies_when

            # Check regime match
            regime_match = True
            if "regime" in applies:
                for key, value in applies["regime"].items():
                    if regime.get(key) != value:
                        regime_match = False
                        break

            # Check strategy match
            strategy_match = True
            if "strategy" in applies:
                if applies["strategy"] != strategy:
                    strategy_match = False

            if regime_match and strategy_match:
                applicable.append(insight)

        return sorted(applicable, key=lambda i: i.confidence, reverse=True)

    async def update_insight_evidence(
        self,
        insight_id: str,
        trade_id: str,
        success: bool,
    ) -> Optional[StrategyInsight]:
        """Update insight with new evidence from a trade.

        Args:
            insight_id: Insight to update
            trade_id: Trade that provides evidence
            success: Whether following the insight led to success

        Returns:
            Updated StrategyInsight or None
        """
        if not self.redis.is_connected:
            return None

        try:
            insight = await self.get_insight(insight_id)
            if not insight:
                return None

            # Add trade to supporting evidence
            if trade_id not in insight.supporting_trades:
                insight.supporting_trades.append(trade_id)
            insight.sample_size = len(insight.supporting_trades)
            insight.times_applied += 1
            insight.updated_at = datetime.now()

            # Update success rate
            if insight.success_rate is None:
                insight.success_rate = 1.0 if success else 0.0
            else:
                # Exponential moving average
                alpha = 0.1
                insight.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * insight.success_rate

            # Update confidence based on sample size and success rate
            insight.confidence = min(
                0.95,
                insight.success_rate * (1 - 1 / (1 + insight.sample_size * 0.1))
            )

            # Store updated insight
            client = self.redis.client
            insight_key = f"{self.SEMANTIC_PREFIX}:{insight_id}"
            await client.set(insight_key, self._serialize(insight), ex=self.INSIGHT_TTL)

            return insight

        except Exception as e:
            logger.error(f"Failed to update insight evidence: {e}")
            return None

    # =========================================================================
    # PROCEDURAL MEMORY (Strategy Parameters)
    # =========================================================================

    async def get_parameter(
        self,
        agent: str,
        parameter: str,
    ) -> Optional[ParameterState]:
        """Get current state of a parameter.

        Args:
            agent: Agent name
            parameter: Parameter name

        Returns:
            ParameterState or None
        """
        if not self.redis.is_connected:
            return None

        try:
            client = self.redis.client
            param_key = f"{self.PROCEDURAL_PREFIX}:{agent}:{parameter}"
            data = await client.get(param_key)
            if data:
                return ParameterState(**self._deserialize(data))
            return None

        except Exception as e:
            logger.error(f"Failed to get parameter: {e}")
            return None

    async def set_parameter(
        self,
        agent: str,
        parameter: str,
        value: Any,
        reason: str,
        default_value: Optional[Any] = None,
        min_bound: Optional[Any] = None,
        max_bound: Optional[Any] = None,
    ) -> ParameterState:
        """Set or update a parameter value.

        Args:
            agent: Agent name
            parameter: Parameter name
            value: New value
            reason: Reason for change
            default_value: Default value (for new parameters)
            min_bound: Minimum allowed value
            max_bound: Maximum allowed value

        Returns:
            Updated ParameterState
        """
        # Get existing or create new
        existing = await self.get_parameter(agent, parameter)

        if existing:
            # Record in history before updating
            existing.history.append({
                "value": existing.current_value,
                "timestamp": existing.last_updated.isoformat(),
                "reason": existing.update_reason,
            })

            # Keep only last 50 history entries
            existing.history = existing.history[-50:]

            existing.current_value = value
            existing.last_updated = datetime.now()
            existing.update_reason = reason

            param_state = existing
        else:
            param_state = ParameterState(
                agent=agent,
                parameter=parameter,
                current_value=value,
                default_value=default_value or value,
                min_bound=min_bound,
                max_bound=max_bound,
                last_updated=datetime.now(),
                update_reason=reason,
            )

        if not self.redis.is_connected:
            return param_state

        try:
            client = self.redis.client
            param_key = f"{self.PROCEDURAL_PREFIX}:{agent}:{parameter}"

            # Store parameter
            await client.set(param_key, self._serialize(param_state))

            # Add to index
            await client.sadd(f"{self.PROCEDURAL_PREFIX}:index:{agent}", parameter)

            logger.info(f"Parameter updated: {agent}.{parameter} = {value} ({reason})")
            return param_state

        except Exception as e:
            logger.error(f"Failed to set parameter: {e}")
            return param_state

    async def get_all_parameters(self, agent: str) -> Dict[str, ParameterState]:
        """Get all parameters for an agent.

        Args:
            agent: Agent name

        Returns:
            Dictionary of parameter name to ParameterState
        """
        if not self.redis.is_connected:
            return {}

        try:
            client = self.redis.client
            param_names = await client.smembers(f"{self.PROCEDURAL_PREFIX}:index:{agent}")

            params = {}
            for param_name in param_names:
                if isinstance(param_name, bytes):
                    param_name = param_name.decode("utf-8")
                param = await self.get_parameter(agent, param_name)
                if param:
                    params[param_name] = param

            return params

        except Exception as e:
            logger.error(f"Failed to get all parameters for {agent}: {e}")
            return {}

    async def record_parameter_evolution(self, evolution: ParameterEvolution) -> str:
        """Record a parameter evolution event.

        Args:
            evolution: ParameterEvolution record

        Returns:
            evolution_id
        """
        if not self.redis.is_connected:
            return evolution.evolution_id

        try:
            client = self.redis.client

            # Add to evolution list
            await client.lpush(
                f"{self.PROCEDURAL_PREFIX}:evolution",
                self._serialize(evolution),
            )

            # Trim to keep only last 1000 evolutions
            await client.ltrim(f"{self.PROCEDURAL_PREFIX}:evolution", 0, 999)

            logger.info(
                f"Parameter evolution recorded: {evolution.agent}.{evolution.parameter} "
                f"{evolution.old_value} -> {evolution.new_value}"
            )
            return evolution.evolution_id

        except Exception as e:
            logger.error(f"Failed to record evolution: {e}")
            return evolution.evolution_id

    async def get_recent_evolutions(self, limit: int = 50) -> List[ParameterEvolution]:
        """Get recent parameter evolutions.

        Args:
            limit: Maximum number to return

        Returns:
            List of ParameterEvolution records
        """
        if not self.redis.is_connected:
            return []

        try:
            client = self.redis.client
            data_list = await client.lrange(
                f"{self.PROCEDURAL_PREFIX}:evolution",
                0,
                limit - 1,
            )

            evolutions = []
            for data in data_list:
                evolutions.append(ParameterEvolution(**self._deserialize(data)))

            return evolutions

        except Exception as e:
            logger.error(f"Failed to get recent evolutions: {e}")
            return []

    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================

    async def get_performance_by_strategy(
        self,
        lookback_days: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics grouped by strategy.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary of strategy name to performance metrics
        """
        since = datetime.now() - timedelta(days=lookback_days)
        trades = await self.get_trades_since(since)

        # Filter to closed trades
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]

        # Group by strategy
        by_strategy: Dict[str, List[TradeRecord]] = {}
        for trade in closed_trades:
            strategy = trade.entry_strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(trade)

        # Calculate metrics
        metrics = {}
        for strategy, strategy_trades in by_strategy.items():
            wins = [t for t in strategy_trades if t.pnl and t.pnl > 0]
            losses = [t for t in strategy_trades if t.pnl and t.pnl <= 0]

            total_pnl = sum(t.pnl for t in strategy_trades if t.pnl)
            avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

            metrics[strategy] = {
                "total_trades": len(strategy_trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(strategy_trades) if strategy_trades else 0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
                "avg_score": sum(t.score for t in strategy_trades if t.score) / len(strategy_trades) if strategy_trades else 0,
            }

        return metrics

    async def get_performance_by_regime(
        self,
        lookback_days: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics grouped by market regime.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary of regime hash to performance metrics
        """
        since = datetime.now() - timedelta(days=lookback_days)
        trades = await self.get_trades_since(since)

        # Filter to closed trades
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]

        # Group by regime
        by_regime: Dict[str, List[TradeRecord]] = {}
        for trade in closed_trades:
            regime = trade.entry_regime
            # Create a simple key from regime
            regime_key = f"{regime.get('volatility', 'unknown')}_{regime.get('trend', 'unknown')}"
            if regime_key not in by_regime:
                by_regime[regime_key] = []
            by_regime[regime_key].append(trade)

        # Calculate metrics
        metrics = {}
        for regime_key, regime_trades in by_regime.items():
            wins = [t for t in regime_trades if t.pnl and t.pnl > 0]
            losses = [t for t in regime_trades if t.pnl and t.pnl <= 0]

            total_pnl = sum(t.pnl for t in regime_trades if t.pnl)

            metrics[regime_key] = {
                "total_trades": len(regime_trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(regime_trades) if regime_trades else 0,
                "total_pnl": total_pnl,
            }

        return metrics

    async def get_win_rate(
        self,
        strategy: Optional[str] = None,
        lookback_trades: int = 20,
    ) -> float:
        """Calculate recent win rate.

        Args:
            strategy: Optional strategy filter
            lookback_trades: Number of recent trades to consider

        Returns:
            Win rate as float (0-1)
        """
        if strategy:
            trades = await self.get_trades_by_strategy(strategy, limit=lookback_trades)
        else:
            trades = await self.get_trades_since(
                datetime.now() - timedelta(days=30)
            )
            trades = trades[:lookback_trades]

        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        if not closed_trades:
            return 0.5  # Default to 50% if no history

        wins = [t for t in closed_trades if t.pnl and t.pnl > 0]
        return len(wins) / len(closed_trades)

    # =========================================================================
    # DAILY REPORTS
    # =========================================================================

    async def store_daily_report(self, report: DailyLearningReport) -> str:
        """Store a daily learning report.

        Args:
            report: DailyLearningReport to store

        Returns:
            report_id
        """
        if not self.redis.is_connected:
            return report.report_id

        try:
            client = self.redis.client
            report_key = f"{self.METRICS_PREFIX}:daily:{report.date}"

            await client.set(
                report_key,
                self._serialize(report),
                ex=90 * 24 * 3600,  # 90 days
            )

            logger.info(f"Daily report stored: {report.date}")
            return report.report_id

        except Exception as e:
            logger.error(f"Failed to store daily report: {e}")
            return report.report_id

    async def get_daily_report(self, date: str) -> Optional[DailyLearningReport]:
        """Get a daily report by date.

        Args:
            date: Date string YYYY-MM-DD

        Returns:
            DailyLearningReport or None
        """
        if not self.redis.is_connected:
            return None

        try:
            client = self.redis.client
            report_key = f"{self.METRICS_PREFIX}:daily:{date}"
            data = await client.get(report_key)
            if data:
                return DailyLearningReport(**self._deserialize(data))
            return None

        except Exception as e:
            logger.error(f"Failed to get daily report: {e}")
            return None
