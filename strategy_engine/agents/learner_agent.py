"""Learner Agent for Pattern Extraction and Parameter Evolution.

Extracts patterns from trade history and evolves strategy parameters
autonomously within safe bounds.

Responsibilities:
1. Identify patterns in trade outcomes
2. Generate actionable insights
3. Propose parameter adjustments
4. Apply changes within bounds
5. Track evolution history

Runs daily via ConsolidationLoop + triggered by significant events.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from .base_agent import BaseAgent
from ..models.memory import (
    TradeRecord,
    TradeStatus,
    StrategyInsight,
    ParameterState,
    ParameterEvolution,
    DailyLearningReport,
)


class LearnerAgent(BaseAgent):
    """Agent that extracts patterns and evolves parameters.

    The core of the self-evolving RL system. Learns from trade
    outcomes and adjusts strategy parameters autonomously.
    """

    name = "learner_agent"
    stage_name = "Learning & Evolution"
    model_name = "learner-v1"

    # Parameter bounds for autonomous evolution
    # Format: {agent: {parameter: (min, max)}}
    PARAMETER_BOUNDS = {
        "portfolio_manager": {
            "base_confidence_threshold": (0.4, 0.8),
            "max_portfolio_exposure": (0.3, 0.6),
            "max_single_position": (0.05, 0.15),
        },
        "mean_reversion": {
            "rsi_oversold": (20, 35),
            "rsi_overbought": (65, 80),
            "bb_std": (1.5, 2.5),
            "max_position_pct": (0.03, 0.08),
        },
        "trend_following": {
            "atr_multiplier": (1.5, 3.0),
            "channel_period": (15, 30),
            "max_position_pct": (0.05, 0.15),
        },
        "turtle_trading": {
            "stop_atr_mult": (1.5, 3.0),
            "risk_per_trade": (0.005, 0.02),
            "system1_entry": (15, 25),
            "system2_entry": (45, 60),
        },
        "risk_manager": {
            "max_leverage": (5, 20),
            "max_position_pct": (0.05, 0.15),
        },
    }

    # Minimum sample sizes for learning
    MIN_TRADES_FOR_PATTERN = 5
    MIN_TRADES_FOR_EVOLUTION = 10

    # Evolution constraints
    MAX_CHANGE_PER_CYCLE = 0.15  # Max 15% change per evolution cycle
    CONFIDENCE_FOR_EVOLUTION = 0.7  # Min insight confidence to evolve

    DEFAULT_CONFIG = {
        "min_trades_for_pattern": 5,
        "min_trades_for_evolution": 10,
        "max_change_per_cycle": 0.15,
        "confidence_for_evolution": 0.7,
        "lookback_days": 30,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        trade_memory=None,
        llm_analyzer=None,
    ):
        """Initialize Learner Agent.

        Args:
            config: Agent configuration
            trade_memory: TradeMemoryService for accessing history
            llm_analyzer: LLM for pattern extraction
        """
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        self.trade_memory = trade_memory
        self.llm = llm_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process learning cycle.

        Args:
            context: Must contain:
                - trades: List of recent TradeRecords
                - current_parameters: Current parameter states

        Returns:
            Dictionary with:
                - insights: Extracted patterns
                - evolutions: Parameter changes
                - report: Learning summary
        """
        trades = context.get("trades", [])
        current_params = context.get("current_parameters", {})

        # Run consolidation learning
        result = await self.consolidate_learning(trades, current_params)

        return {
            **result,
            "confidence": 1.0 if result.get("insights") else 0.5,
        }

    async def consolidate_learning(
        self,
        trades: List[TradeRecord],
        current_parameters: Optional[Dict[str, Dict[str, ParameterState]]] = None,
    ) -> Dict[str, Any]:
        """Run daily learning consolidation.

        Args:
            trades: Recent trades to analyze
            current_parameters: Current parameter states by agent

        Returns:
            Dictionary with insights, evolutions, and report
        """
        if not trades:
            return {
                "insights": [],
                "evolutions": [],
                "report": None,
                "explanation": "No trades to analyze",
            }

        # Filter to closed trades only
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]

        if len(closed_trades) < self.config["min_trades_for_pattern"]:
            return {
                "insights": [],
                "evolutions": [],
                "report": None,
                "explanation": f"Need {self.config['min_trades_for_pattern']} trades, have {len(closed_trades)}",
            }

        logger.info(f"Consolidating learning from {len(closed_trades)} trades")

        # Step 1: Calculate performance metrics
        metrics = self._calculate_metrics(closed_trades)

        # Step 2: Extract patterns
        insights = await self._extract_patterns(closed_trades, metrics)

        # Step 3: Propose parameter changes
        evolutions = []
        if len(closed_trades) >= self.config["min_trades_for_evolution"]:
            evolutions = await self._propose_parameter_changes(
                insights=insights,
                metrics=metrics,
                current_parameters=current_parameters,
            )

        # Step 4: Generate report
        report = self._generate_report(closed_trades, insights, evolutions)

        return {
            "insights": insights,
            "evolutions": evolutions,
            "report": report,
            "metrics": metrics,
            "explanation": f"Extracted {len(insights)} insights, proposed {len(evolutions)} parameter changes",
        }

    def _calculate_metrics(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Calculate performance metrics from trades.

        Args:
            trades: List of closed trades

        Returns:
            Dictionary of metrics
        """
        if not trades:
            return {}

        # Overall metrics
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades if t.pnl)
        avg_pnl = total_pnl / len(trades) if trades else 0

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # By strategy
        by_strategy: Dict[str, Dict[str, Any]] = {}
        for trade in trades:
            strategy = trade.entry_strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = {"trades": [], "wins": 0, "losses": 0, "pnl": 0}
            by_strategy[strategy]["trades"].append(trade)
            if trade.pnl and trade.pnl > 0:
                by_strategy[strategy]["wins"] += 1
            else:
                by_strategy[strategy]["losses"] += 1
            by_strategy[strategy]["pnl"] += trade.pnl or 0

        for strategy in by_strategy:
            total = len(by_strategy[strategy]["trades"])
            by_strategy[strategy]["win_rate"] = by_strategy[strategy]["wins"] / total if total else 0

        # By regime
        by_regime: Dict[str, Dict[str, Any]] = {}
        for trade in trades:
            regime = trade.entry_regime
            regime_key = f"{regime.get('volatility', 'unknown')}_{regime.get('trend', 'unknown')}"

            if regime_key not in by_regime:
                by_regime[regime_key] = {"trades": [], "wins": 0, "losses": 0, "pnl": 0}
            by_regime[regime_key]["trades"].append(trade)
            if trade.pnl and trade.pnl > 0:
                by_regime[regime_key]["wins"] += 1
            else:
                by_regime[regime_key]["losses"] += 1
            by_regime[regime_key]["pnl"] += trade.pnl or 0

        for regime_key in by_regime:
            total = len(by_regime[regime_key]["trades"])
            by_regime[regime_key]["win_rate"] = by_regime[regime_key]["wins"] / total if total else 0

        # Worst and best trades
        sorted_by_pnl = sorted(trades, key=lambda t: t.pnl or 0)
        worst_trades = sorted_by_pnl[:3]
        best_trades = sorted_by_pnl[-3:][::-1]

        # Average score
        scored_trades = [t for t in trades if t.score is not None]
        avg_score = sum(t.score for t in scored_trades) / len(scored_trades) if scored_trades else 0

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_score": avg_score,
            "by_strategy": by_strategy,
            "by_regime": by_regime,
            "worst_trades": worst_trades,
            "best_trades": best_trades,
        }

    async def _extract_patterns(
        self,
        trades: List[TradeRecord],
        metrics: Dict[str, Any],
    ) -> List[StrategyInsight]:
        """Extract patterns from trade outcomes.

        Args:
            trades: Closed trades
            metrics: Calculated metrics

        Returns:
            List of extracted insights
        """
        insights = []

        # Pattern 1: Strategy-regime performance
        for strategy, data in metrics.get("by_strategy", {}).items():
            win_rate = data.get("win_rate", 0)
            n_trades = len(data.get("trades", []))

            if n_trades >= self.config["min_trades_for_pattern"]:
                # Find regime breakdown for this strategy
                strategy_trades = data["trades"]
                regime_breakdown = {}

                for trade in strategy_trades:
                    regime = trade.entry_regime
                    regime_key = f"{regime.get('volatility', 'unknown')}_{regime.get('trend', 'unknown')}"
                    if regime_key not in regime_breakdown:
                        regime_breakdown[regime_key] = {"wins": 0, "losses": 0}
                    if trade.pnl and trade.pnl > 0:
                        regime_breakdown[regime_key]["wins"] += 1
                    else:
                        regime_breakdown[regime_key]["losses"] += 1

                # Find worst performing regime for this strategy
                for regime_key, regime_data in regime_breakdown.items():
                    total = regime_data["wins"] + regime_data["losses"]
                    if total >= 3:  # Minimum sample
                        regime_win_rate = regime_data["wins"] / total

                        if regime_win_rate < 0.3:  # Very poor performance
                            parts = regime_key.split("_")
                            volatility = parts[0] if len(parts) > 0 else "unknown"
                            trend = parts[1] if len(parts) > 1 else "unknown"

                            insight = StrategyInsight(
                                insight_id=f"pattern_{strategy}_{regime_key}_{uuid.uuid4().hex[:8]}",
                                pattern=f"{strategy}_fails_{regime_key}",
                                description=f"{strategy} strategy performs poorly ({regime_win_rate*100:.0f}% win rate) in {volatility} volatility / {trend} trend conditions",
                                supporting_trades=[t.trade_id for t in strategy_trades if
                                    f"{t.entry_regime.get('volatility')}_{t.entry_regime.get('trend')}" == regime_key],
                                confidence=min(0.9, 0.5 + (total / 20)),  # Higher confidence with more samples
                                sample_size=total,
                                applies_when={
                                    "regime": {"volatility": volatility, "trend": trend},
                                    "strategy": strategy,
                                },
                                recommendation=f"Reduce or skip {strategy} signals when volatility is {volatility} and trend is {trend}",
                                affected_parameters=[f"{strategy}.base_confidence_threshold", f"{strategy}.max_position_pct"],
                            )
                            insights.append(insight)

                        elif regime_win_rate > 0.7:  # Excellent performance
                            parts = regime_key.split("_")
                            volatility = parts[0] if len(parts) > 0 else "unknown"
                            trend = parts[1] if len(parts) > 1 else "unknown"

                            insight = StrategyInsight(
                                insight_id=f"pattern_{strategy}_{regime_key}_good_{uuid.uuid4().hex[:8]}",
                                pattern=f"{strategy}_excels_{regime_key}",
                                description=f"{strategy} strategy excels ({regime_win_rate*100:.0f}% win rate) in {volatility} volatility / {trend} trend conditions",
                                supporting_trades=[t.trade_id for t in strategy_trades if
                                    f"{t.entry_regime.get('volatility')}_{t.entry_regime.get('trend')}" == regime_key],
                                confidence=min(0.9, 0.5 + (total / 20)),
                                sample_size=total,
                                applies_when={
                                    "regime": {"volatility": volatility, "trend": trend},
                                    "strategy": strategy,
                                },
                                recommendation=f"Increase position size for {strategy} signals when volatility is {volatility} and trend is {trend}",
                                affected_parameters=[f"{strategy}.max_position_pct", "portfolio_manager.base_confidence_threshold"],
                            )
                            insights.append(insight)

        # Pattern 2: Overall strategy underperformance
        for strategy, data in metrics.get("by_strategy", {}).items():
            win_rate = data.get("win_rate", 0)
            n_trades = len(data.get("trades", []))

            if n_trades >= self.config["min_trades_for_pattern"] and win_rate < 0.35:
                insight = StrategyInsight(
                    insight_id=f"pattern_{strategy}_underperform_{uuid.uuid4().hex[:8]}",
                    pattern=f"{strategy}_underperforming",
                    description=f"{strategy} strategy is underperforming overall ({win_rate*100:.0f}% win rate over {n_trades} trades)",
                    supporting_trades=[t.trade_id for t in data["trades"]],
                    confidence=min(0.85, 0.4 + (n_trades / 50)),
                    sample_size=n_trades,
                    applies_when={"strategy": strategy},
                    recommendation=f"Tighten confidence threshold for {strategy} strategy or review entry criteria",
                    affected_parameters=[f"portfolio_manager.base_confidence_threshold", f"{strategy}.max_position_pct"],
                )
                insights.append(insight)

        # Pattern 3: Risk management issues
        high_loss_trades = [t for t in trades if t.pnl_pct and t.pnl_pct < -3]
        if len(high_loss_trades) >= 3:
            insight = StrategyInsight(
                insight_id=f"pattern_risk_high_losses_{uuid.uuid4().hex[:8]}",
                pattern="excessive_losses",
                description=f"Found {len(high_loss_trades)} trades with >3% losses - stop losses may be too wide",
                supporting_trades=[t.trade_id for t in high_loss_trades],
                confidence=0.75,
                sample_size=len(high_loss_trades),
                applies_when={},
                recommendation="Tighten stop loss distances (reduce ATR multiplier)",
                affected_parameters=["turtle_trading.stop_atr_mult", "trend_following.atr_multiplier"],
            )
            insights.append(insight)

        # Use LLM for deeper pattern extraction if available
        if self.llm and len(trades) >= 10:
            llm_insights = await self._llm_extract_patterns(trades, metrics)
            insights.extend(llm_insights)

        logger.info(f"Extracted {len(insights)} patterns from {len(trades)} trades")
        return insights

    async def _llm_extract_patterns(
        self,
        trades: List[TradeRecord],
        metrics: Dict[str, Any],
    ) -> List[StrategyInsight]:
        """Use LLM to extract deeper patterns.

        Args:
            trades: Closed trades
            metrics: Calculated metrics

        Returns:
            List of LLM-extracted insights
        """
        # Build prompt with trade summary
        strategy_breakdown = ""
        for strategy, data in metrics.get("by_strategy", {}).items():
            strategy_breakdown += f"- {strategy}: {data['win_rate']*100:.0f}% win rate, {len(data['trades'])} trades, ${data['pnl']:.2f} P&L\n"

        regime_breakdown = ""
        for regime, data in metrics.get("by_regime", {}).items():
            regime_breakdown += f"- {regime}: {data['win_rate']*100:.0f}% win rate, {len(data['trades'])} trades\n"

        worst_trades = ""
        for trade in metrics.get("worst_trades", [])[:3]:
            worst_trades += f"- {trade.entry_strategy}: {trade.pnl_pct:+.2f}%, entry regime: {trade.entry_regime}\n"

        best_trades = ""
        for trade in metrics.get("best_trades", [])[:3]:
            best_trades += f"- {trade.entry_strategy}: {trade.pnl_pct:+.2f}%, entry regime: {trade.entry_regime}\n"

        prompt = f"""
Analyze {len(trades)} recent trades to identify 2-3 specific, actionable patterns:

PERFORMANCE SUMMARY:
- Total P&L: ${metrics.get('total_pnl', 0):.2f}
- Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
- Avg Win: ${metrics.get('avg_win', 0):.2f}
- Avg Loss: ${metrics.get('avg_loss', 0):.2f}
- Profit Factor: {metrics.get('profit_factor', 0):.2f}

BY STRATEGY:
{strategy_breakdown}

BY REGIME:
{regime_breakdown}

WORST TRADES:
{worst_trades}

BEST TRADES:
{best_trades}

For each pattern, provide:
PATTERN: [name in snake_case]
DESCRIPTION: [what was observed]
APPLIES WHEN: [specific conditions]
RECOMMENDATION: [what to do differently]
CONFIDENCE: [high/medium/low]

Focus on actionable insights that can improve trading performance.
"""

        try:
            response = await self.llm.analyze_market(
                symbol="BTCUSDT",
                price=0,
                regime={},
                additional_context=prompt,
            )

            # Parse LLM response into insights (simplified parsing)
            insights = []
            # Note: In production, use proper parsing of LLM response
            # For now, we'll just log the LLM suggestions
            logger.info(f"LLM pattern extraction: {response[:200]}...")

            return insights

        except Exception as e:
            logger.error(f"LLM pattern extraction failed: {e}")
            return []

    async def _propose_parameter_changes(
        self,
        insights: List[StrategyInsight],
        metrics: Dict[str, Any],
        current_parameters: Optional[Dict[str, Dict[str, ParameterState]]],
    ) -> List[ParameterEvolution]:
        """Convert insights into parameter adjustments.

        Args:
            insights: Extracted insights
            metrics: Performance metrics
            current_parameters: Current parameter states

        Returns:
            List of proposed parameter evolutions
        """
        evolutions = []

        for insight in insights:
            if insight.confidence < self.config["confidence_for_evolution"]:
                continue

            for affected_param in insight.affected_parameters:
                parts = affected_param.split(".")
                if len(parts) != 2:
                    continue

                agent, param = parts

                # Check if this parameter is within our bounds
                if agent not in self.PARAMETER_BOUNDS:
                    continue
                if param not in self.PARAMETER_BOUNDS[agent]:
                    continue

                bounds = self.PARAMETER_BOUNDS[agent][param]
                min_val, max_val = bounds

                # Get current value
                current_val = None
                if current_parameters and agent in current_parameters:
                    param_state = current_parameters[agent].get(param)
                    if param_state:
                        current_val = param_state.current_value

                if current_val is None:
                    continue  # Can't evolve without current value

                # Determine direction and magnitude of change
                new_val = self._calculate_new_value(
                    insight=insight,
                    current_val=current_val,
                    min_val=min_val,
                    max_val=max_val,
                    metrics=metrics,
                )

                if new_val is None or new_val == current_val:
                    continue

                evolution = ParameterEvolution(
                    evolution_id=f"evo_{agent}_{param}_{uuid.uuid4().hex[:8]}",
                    agent=agent,
                    parameter=param,
                    old_value=current_val,
                    new_value=new_val,
                    reason=insight.description,
                    supporting_insights=[insight.insight_id],
                    expected_impact=insight.recommendation,
                )
                evolutions.append(evolution)

        logger.info(f"Proposed {len(evolutions)} parameter evolutions")
        return evolutions

    def _calculate_new_value(
        self,
        insight: StrategyInsight,
        current_val: Any,
        min_val: Any,
        max_val: Any,
        metrics: Dict[str, Any],
    ) -> Optional[Any]:
        """Calculate new parameter value based on insight.

        Args:
            insight: The driving insight
            current_val: Current parameter value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            metrics: Performance metrics

        Returns:
            New value or None if no change
        """
        pattern = insight.pattern.lower()
        max_change = self.config["max_change_per_cycle"]

        # Determine direction based on pattern
        if "fail" in pattern or "underperform" in pattern or "excessive_loss" in pattern:
            # Tighten (reduce) parameters
            direction = -1
        elif "excel" in pattern or "outperform" in pattern:
            # Loosen (increase) parameters
            direction = 1
        else:
            return None

        # Calculate magnitude based on confidence and sample size
        magnitude = insight.confidence * 0.5  # Base magnitude from confidence
        magnitude *= min(1.0, insight.sample_size / 20)  # Scale by sample size
        magnitude = min(magnitude, max_change)  # Cap at max change

        # Apply change
        if isinstance(current_val, float):
            change = (max_val - min_val) * magnitude * direction
            new_val = current_val + change
            new_val = max(min_val, min(max_val, new_val))
            # Round to reasonable precision
            new_val = round(new_val, 4)
        elif isinstance(current_val, int):
            change = int((max_val - min_val) * magnitude * direction)
            change = max(1, abs(change)) * (1 if direction > 0 else -1)
            new_val = current_val + change
            new_val = max(min_val, min(max_val, new_val))
        else:
            return None

        return new_val

    def _generate_report(
        self,
        trades: List[TradeRecord],
        insights: List[StrategyInsight],
        evolutions: List[ParameterEvolution],
    ) -> DailyLearningReport:
        """Generate daily learning report.

        Args:
            trades: Analyzed trades
            insights: Extracted insights
            evolutions: Proposed evolutions

        Returns:
            DailyLearningReport
        """
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades if t.pnl)

        # Group by strategy
        strategy_perf = {}
        for trade in trades:
            s = trade.entry_strategy
            if s not in strategy_perf:
                strategy_perf[s] = {"trades": 0, "wins": 0, "pnl": 0}
            strategy_perf[s]["trades"] += 1
            if trade.pnl and trade.pnl > 0:
                strategy_perf[s]["wins"] += 1
            strategy_perf[s]["pnl"] += trade.pnl or 0

        # Group by regime
        regime_perf = {}
        for trade in trades:
            r = f"{trade.entry_regime.get('volatility', 'unknown')}_{trade.entry_regime.get('trend', 'unknown')}"
            if r not in regime_perf:
                regime_perf[r] = {"trades": 0, "wins": 0, "pnl": 0}
            regime_perf[r]["trades"] += 1
            if trade.pnl and trade.pnl > 0:
                regime_perf[r]["wins"] += 1
            regime_perf[r]["pnl"] += trade.pnl or 0

        # Key learnings
        key_learnings = [i.description for i in insights[:3]]

        # Summary
        summary_parts = [
            f"Analyzed {len(trades)} trades.",
            f"Win rate: {len(wins)/len(trades)*100:.0f}%." if trades else "",
            f"Net P&L: ${total_pnl:.2f}.",
            f"Extracted {len(insights)} patterns.",
            f"Proposed {len(evolutions)} parameter changes.",
        ]
        summary = " ".join(p for p in summary_parts if p)

        return DailyLearningReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}",
            date=datetime.now().strftime("%Y-%m-%d"),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=total_pnl,
            strategy_performance=strategy_perf,
            regime_performance=regime_perf,
            new_insights=[i.insight_id for i in insights],
            parameter_changes=[e.evolution_id for e in evolutions],
            summary=summary,
            key_learnings=key_learnings,
        )

    async def apply_evolutions(
        self,
        evolutions: List[ParameterEvolution],
        trade_memory,
    ) -> List[ParameterEvolution]:
        """Apply parameter evolutions to the system.

        Args:
            evolutions: Evolutions to apply
            trade_memory: TradeMemoryService for persistence

        Returns:
            List of successfully applied evolutions
        """
        applied = []

        for evolution in evolutions:
            try:
                # Store in procedural memory
                await trade_memory.set_parameter(
                    agent=evolution.agent,
                    parameter=evolution.parameter,
                    value=evolution.new_value,
                    reason=evolution.reason,
                    min_bound=self.PARAMETER_BOUNDS.get(evolution.agent, {}).get(evolution.parameter, (None, None))[0],
                    max_bound=self.PARAMETER_BOUNDS.get(evolution.agent, {}).get(evolution.parameter, (None, None))[1],
                )

                # Record evolution
                await trade_memory.record_parameter_evolution(evolution)

                logger.info(
                    f"Applied evolution: {evolution.agent}.{evolution.parameter} "
                    f"{evolution.old_value} -> {evolution.new_value}"
                )
                applied.append(evolution)

            except Exception as e:
                logger.error(f"Failed to apply evolution {evolution.evolution_id}: {e}")

        return applied
