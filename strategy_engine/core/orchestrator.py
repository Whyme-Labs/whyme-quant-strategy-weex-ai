"""Agent Orchestrator for multi-agent trading.

Coordinates multiple AI agents to make collaborative trading decisions.

Architecture (based on research insights):
- Regime Detector determines whether to use Mean Reversion or Trend Following
- "Some regimes reward trend following. Others reward mean reversion."
- Running both strategies across different regimes smooths returns and reduces drawdowns.
- AlphaGenerator provides signal confirmation via multi-indicator aggregation
"""

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from loguru import logger

from .base import Signal, SignalAction
from ..agents.base_agent import BaseAgent
from ai_logging import get_ai_logger

if TYPE_CHECKING:
    from ..services.alpha_generator import AlphaGenerator
    from ..services.edge_scanner import EdgeScanner
    from ..services.key_level_detector import KeyLevelDetector
    from ..services.smc_detector import SMCDetector
    from ..services.llm_signal_validator import LLMSignalValidator, LLMDecision


class AgentOrchestrator:
    """Orchestrates multiple AI agents for trading decisions.

    The orchestrator manages the flow of information between agents:
    1. Regime Detector classifies market conditions
    2. Routes to appropriate strategy:
       - Mean Reversion (high win rate, fade extremes)
       - Trend Following (low win rate, big winners)
    3. AlphaGenerator confirms signal direction via indicator aggregation
    4. EdgeScanner provides additional edge-based confirmation
    5. Risk Manager evaluates and adjusts the proposal
    6. Execution Agent optimizes order execution
    """

    def __init__(
        self,
        config: Dict[str, Any],
        alpha_generator: Optional["AlphaGenerator"] = None,
        edge_scanner: Optional["EdgeScanner"] = None,
        key_level_detector: Optional["KeyLevelDetector"] = None,
        smc_detector: Optional["SMCDetector"] = None,
        llm_validator: Optional["LLMSignalValidator"] = None,
    ):
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration
            alpha_generator: Optional AlphaGenerator for signal confirmation
            edge_scanner: Optional EdgeScanner for edge-based confirmation
            key_level_detector: Optional KeyLevelDetector for S/R levels
            smc_detector: Optional SMCDetector for Smart Money Concepts
            llm_validator: Optional LLMSignalValidator for AI-powered signal validation
        """
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.ai_logger = get_ai_logger()
        self._running = False
        self.last_regime = None
        self.last_strategy_analysis: Dict[str, list] = {}  # Track per-symbol strategy analysis
        self.alpha_generator = alpha_generator
        self.edge_scanner = edge_scanner
        self.key_level_detector = key_level_detector
        self.smc_detector = smc_detector
        self.llm_validator = llm_validator  # LLM Signal Validator with full override authority
        self.last_alpha_signal = None  # Track last alpha signal for logging
        self.last_edge_signals = []  # Track last edge signals for logging
        self.last_key_levels = None  # Track last key levels for logging
        self.last_smc_data = None  # Track last SMC data for logging
        self.last_llm_validation = None  # Track last LLM validation result

    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the orchestrator.

        Args:
            name: Agent identifier
            agent: Agent instance
        """
        self.agents[name] = agent
        logger.info(f"Registered agent: {name} ({agent.__class__.__name__})")

    async def process_market_data(
        self,
        market_data: Dict[str, Any],
    ) -> Optional[Signal]:
        """Process market data through the regime-based agent pipeline.

        Flow:
        1. Regime Detector → classifies market (trend/mean-reversion/neutral)
        2. Route to appropriate strategy agent based on regime
        3. Risk Manager → evaluates and sizes position
        4. Execution Agent → optimizes order execution

        Args:
            market_data: Current market data

        Returns:
            Trading signal if agents agree, None otherwise
        """
        account_info = market_data.get("account_info", {})
        symbol = market_data.get("symbol", "BTCUSDT")
        context = {
            "symbol": symbol,  # IMPORTANT: Symbol must be in context for strategy tracking
            "market_data": market_data,
            "timestamp": market_data.get("timestamp"),
            "account_info": account_info,
            "account": {
                "available_balance": account_info.get("available", 0),
                "equity": account_info.get("equity", 0),
                "position_value": account_info.get("usedMargin", 0),
            },
        }

        # Stage 0: Key Level Detection (support/resistance)
        if self.key_level_detector:
            try:
                symbol = market_data.get("symbol", "BTCUSDT")
                current_price = market_data.get("price", 0)
                key_levels = await self.key_level_detector.detect_all_levels(
                    symbol=symbol,
                    candles_1h=market_data.get("candles_1h"),
                    candles_4h=market_data.get("candles_4h"),
                    candles_1d=market_data.get("candles_1d"),
                )
                context["key_levels"] = key_levels
                self.last_key_levels = key_levels

                # Get nearest S/R for quick reference
                if current_price > 0:
                    nearest = await self.key_level_detector.get_nearest_levels(
                        symbol, current_price, count=3
                    )
                    context["nearest_support"] = nearest.get("supports", [{}])[0] if nearest.get("supports") else None
                    context["nearest_resistance"] = nearest.get("resistances", [{}])[0] if nearest.get("resistances") else None

                    # Log key level summary
                    level_summary = self.key_level_detector.get_level_summary(symbol, current_price)
                    if level_summary.get("has_levels"):
                        logger.debug(
                            f"Key levels: {level_summary.get('total_levels', 0)} levels | "
                            f"Nearest S: {level_summary.get('distance_to_support_pct', 0):.2f}% | "
                            f"Nearest R: {level_summary.get('distance_to_resistance_pct', 0):.2f}%"
                        )
            except Exception as e:
                logger.warning(f"Key level detection failed: {e}")
                context["key_levels"] = None

        # Stage 0.5: SMC Detection (Smart Money Concepts)
        if self.smc_detector:
            try:
                symbol = market_data.get("symbol", "BTCUSDT")
                current_price = market_data.get("price", 0)
                smc_data = await self.smc_detector.detect_all(
                    symbol=symbol,
                    candles_1h=market_data.get("candles_1h"),
                    candles_4h=market_data.get("candles_4h"),
                    candles_1d=market_data.get("candles_1d"),
                )
                context["smc"] = smc_data
                self.last_smc_data = smc_data

                # Get SMC summary for quick reference
                if current_price > 0:
                    smc_summary = self.smc_detector.get_smc_summary(symbol, current_price)
                    context["smc_summary"] = smc_summary

                    # Log comprehensive SMC information
                    trend = smc_summary.get("trend", "unknown")
                    prem_disc = smc_summary.get("premium_discount", {})
                    zone = prem_disc.get("current_zone", "unknown") if prem_disc else "unknown"
                    distance_pct = prem_disc.get("distance_pct", 0) if prem_disc else 0
                    active_obs = smc_summary.get("active_ob_count", 0)
                    active_fvgs = smc_summary.get("active_fvg_count", 0)
                    buy_liq = smc_summary.get("buy_liquidity_pools", 0)
                    sell_liq = smc_summary.get("sell_liquidity_pools", 0)
                    structure = smc_summary.get("structure_summary", {})

                    logger.info(
                        f"SMC [{symbol}]: trend={trend}, zone={zone} ({distance_pct:+.2f}%), "
                        f"OBs={active_obs}, FVGs={active_fvgs}, liq_pools=buy:{buy_liq}/sell:{sell_liq}, "
                        f"structure=HH:{structure.get('hh_count', 0)}/HL:{structure.get('hl_count', 0)}/"
                        f"LH:{structure.get('lh_count', 0)}/LL:{structure.get('ll_count', 0)}"
                    )

                    # Log order blocks detail
                    if smc_data.get("order_blocks"):
                        for ob in smc_data["order_blocks"][:3]:
                            logger.debug(
                                f"  OB: {ob.get('direction')} {ob.get('timeframe')} | "
                                f"zone={ob.get('price_low'):.2f}-{ob.get('price_high'):.2f} | "
                                f"impulse={ob.get('impulse_size_pct'):.2f}% | "
                                f"strength={ob.get('strength')}"
                            )

                    # Log FVG detail
                    if smc_data.get("fair_value_gaps"):
                        for fvg in smc_data["fair_value_gaps"][:3]:
                            logger.debug(
                                f"  FVG: {fvg.get('direction')} {fvg.get('timeframe')} | "
                                f"gap={fvg.get('gap_low'):.2f}-{fvg.get('gap_high'):.2f} | "
                                f"size={fvg.get('gap_size_pct'):.2f}% | "
                                f"filled={fvg.get('filled')}"
                            )

                    # Log BOS/CHoCH events
                    if smc_data.get("bos_events"):
                        for bos in smc_data["bos_events"][-2:]:
                            logger.debug(
                                f"  BOS: {bos.get('direction')} {bos.get('timeframe')} | "
                                f"break_price={bos.get('break_price'):.2f} | "
                                f"confirmed={bos.get('confirmed')}"
                            )

                    if smc_data.get("choch_events"):
                        for choch in smc_data["choch_events"][-2:]:
                            logger.info(
                                f"  CHoCH: {choch.get('direction')} {choch.get('timeframe')} | "
                                f"choch_price={choch.get('choch_price'):.2f} | "
                                f"old_trend={choch.get('old_trend')}"
                            )

                    # Log liquidity pools
                    if smc_data.get("liquidity_pools"):
                        for pool in smc_data["liquidity_pools"][:3]:
                            logger.debug(
                                f"  LIQ: {pool.get('side')}-side {pool.get('timeframe')} | "
                                f"level={pool.get('price_level'):.2f} | "
                                f"source={pool.get('source')} | "
                                f"touches={pool.get('num_touches')} | "
                                f"swept={pool.get('swept')}"
                            )

                    # Log inducements (potential stop hunts)
                    if smc_data.get("inducements"):
                        for ind in smc_data["inducements"]:
                            logger.info(
                                f"  INDUCEMENT: {ind.get('direction')} {ind.get('timeframe')} | "
                                f"level={ind.get('inducement_level'):.2f} | "
                                f"sweep={ind.get('sweep_low'):.2f}-{ind.get('sweep_high'):.2f} | "
                                f"reversal_confirmed={ind.get('reversal_confirmed')}"
                            )

            except Exception as e:
                logger.warning(f"SMC detection failed: {e}")
                context["smc"] = None
                context["smc_summary"] = None

        # Stage 1: Regime Detection (primary routing decision)
        if "regime_detector" in self.agents:
            regime_result = await self._run_agent("regime_detector", context)
            context["regime"] = regime_result.get("regime", {})
            context["recommended_strategy"] = regime_result.get("recommended_strategy", "neutral")
            context["regime_confidence"] = regime_result.get("confidence", 0)
            self.last_regime = regime_result
            logger.info(f"Regime: {context['regime']}, Recommended: {context['recommended_strategy']} (confidence: {context['regime_confidence']:.2f})")
        else:
            # Fallback: use market analyst for basic analysis
            if "market_analyst" in self.agents:
                analysis = await self._run_agent("market_analyst", context)
                context["market_analysis"] = analysis
                context["recommended_strategy"] = "trend_following"  # Default
                logger.debug(f"Market analysis (fallback): {analysis}")

        # Stage 2: Strategy Generation (regime-based routing)
        strategy_result = await self._generate_strategy_by_regime(context)

        if not strategy_result or not strategy_result.get("signal"):
            logger.debug(f"No signal from {context.get('recommended_strategy', 'unknown')} strategy")
            return None

        context["strategy_proposal"] = self._convert_signal_to_proposal(strategy_result)
        logger.info(f"Strategy proposal: {context['strategy_proposal']}")

        # Stage 2.5: Alpha Generator Confirmation (boost only, no rejection)
        if self.alpha_generator:
            alpha_signal = await self._validate_with_alpha_generator(
                context["strategy_proposal"], market_data.get("symbol", "BTCUSDT")
            )
            context["alpha_confirmation"] = alpha_signal
            self.last_alpha_signal = alpha_signal

            if alpha_signal:
                strategy_direction = context["strategy_proposal"].get("action", "hold")
                alpha_direction = alpha_signal.direction

                # Boost confidence if alpha agrees (no rejection if disagrees)
                if (strategy_direction == "buy" and alpha_direction == "LONG") or \
                   (strategy_direction == "sell" and alpha_direction == "SHORT"):
                    original_confidence = context["strategy_proposal"].get("confidence", 0.5)
                    boost = min(0.15, abs(alpha_signal.alpha) * 0.2)
                    context["strategy_proposal"]["confidence"] = min(1.0, original_confidence + boost)
                    logger.debug(
                        f"Alpha confirmation boost: {original_confidence:.2f} -> "
                        f"{context['strategy_proposal']['confidence']:.2f}"
                    )
                else:
                    # Log disagreement but don't reject
                    logger.debug(
                        f"Alpha disagrees: Strategy={strategy_direction}, "
                        f"Alpha={alpha_direction} ({alpha_signal.alpha:+.2f}) - continuing anyway"
                    )

        # Stage 2.6: Edge Scanner Confirmation (statistical edge validation)
        if self.edge_scanner:
            edge_signals = await self._validate_with_edge_scanner(
                context["strategy_proposal"], market_data.get("symbol", "BTCUSDT")
            )
            context["edge_confirmation"] = edge_signals
            self.last_edge_signals = edge_signals

            if edge_signals:
                # Check if any edge signal aligns with strategy
                strategy_direction = context["strategy_proposal"].get("action", "hold")

                for edge_signal in edge_signals:
                    edge_side = edge_signal.side  # "long" or "short"

                    # Boost confidence if edge agrees with strategy direction
                    if (strategy_direction == "buy" and edge_side == "long") or \
                       (strategy_direction == "sell" and edge_side == "short"):
                        original_confidence = context["strategy_proposal"].get("confidence", 0.5)
                        # Boost based on edge expectancy (stronger edges = bigger boost)
                        edge_expectancy = edge_signal.edge.expectancy if edge_signal.edge else 0
                        boost = min(0.10, edge_expectancy * 0.15)
                        context["strategy_proposal"]["confidence"] = min(1.0, original_confidence + boost)
                        logger.debug(
                            f"Edge confirmation boost ({edge_signal.edge_id}): "
                            f"{original_confidence:.2f} -> {context['strategy_proposal']['confidence']:.2f}"
                        )
                        break  # Only use first matching edge

        # Stage 2.7: LLM Signal Validation (AI-powered decision with FULL override authority)
        # The LLM can:
        # 1. APPROVE signals - signal proceeds to Portfolio Manager
        # 2. REJECT signals - signal is blocked before Portfolio Manager
        # 3. ADJUST confidence - boost or reduce based on LLM analysis
        if self.llm_validator and context.get("strategy_proposal"):
            from ..services.llm_signal_validator import LLMDecision

            # Get recent trades for context (from trade memory if available)
            recent_trades = []  # Could be populated from trade_memory if needed

            llm_result = await self.llm_validator.validate_signal(
                signal=context["strategy_proposal"],
                market_context=context,
                alpha_result=context.get("alpha_confirmation"),
                edge_result=context.get("edge_confirmation"),
                key_levels=context.get("key_levels"),
                smc_data=context.get("smc"),
                recent_trades=recent_trades,
            )
            context["llm_validation"] = llm_result
            self.last_llm_validation = llm_result

            # Apply LLM decision
            if llm_result.decision == LLMDecision.REJECT:
                # LLM rejected the signal - mark it for Portfolio Manager to skip
                context["strategy_proposal"]["llm_rejected"] = True
                context["strategy_proposal"]["llm_rejection_reason"] = llm_result.reasoning
                logger.info(
                    f"LLM REJECTED {market_data.get('symbol', 'BTCUSDT')} "
                    f"{context['strategy_proposal'].get('action', 'unknown').upper()}: "
                    f"{llm_result.reasoning[:100]}..."
                )
            elif llm_result.decision in [LLMDecision.APPROVE, LLMDecision.DEFER]:
                # LLM approved or deferred - apply confidence adjustment
                original_confidence = context["strategy_proposal"].get("confidence", 0.5)
                new_confidence = max(0.0, min(1.0, original_confidence + llm_result.confidence_adjustment))
                context["strategy_proposal"]["confidence"] = new_confidence
                context["strategy_proposal"]["llm_reasoning"] = llm_result.reasoning
                context["strategy_proposal"]["llm_context"] = llm_result.market_context
                context["strategy_proposal"]["llm_risk"] = llm_result.risk_assessment

                if llm_result.confidence_adjustment != 0:
                    logger.info(
                        f"LLM confidence adjustment for {market_data.get('symbol', 'BTCUSDT')}: "
                        f"{original_confidence:.2f} -> {new_confidence:.2f} "
                        f"({llm_result.confidence_adjustment:+.2f})"
                    )
                if llm_result.decision == LLMDecision.APPROVE:
                    logger.info(
                        f"LLM APPROVED {market_data.get('symbol', 'BTCUSDT')} "
                        f"{context['strategy_proposal'].get('action', 'unknown').upper()}: "
                        f"{llm_result.reasoning[:100]}..."
                    )

        # Stage 3: Portfolio Management (dynamic confidence threshold)
        # This is the bridge between signals and execution
        if "portfolio_manager" in self.agents:
            portfolio_result = await self._run_agent("portfolio_manager", context)
            context["portfolio_decision"] = portfolio_result

            if not portfolio_result.get("approved", False):
                rejection_reason = portfolio_result.get('reasoning', 'Unknown rejection reason')
                logger.info(
                    f"Trade rejected by portfolio manager: {rejection_reason}. "
                    f"Signal confidence: {portfolio_result.get('signal_confidence', 0):.2f}, "
                    f"Threshold: {portfolio_result.get('confidence_threshold', 0):.2f}"
                )

                # Stage 3.5: LLM Override Review for rejected signals
                # LLM can override Portfolio Manager rejection if it sees opportunity
                if self.llm_validator and context.get("strategy_proposal"):
                    from ..services.llm_signal_validator import LLMDecision

                    # Don't review if LLM already rejected
                    if not context["strategy_proposal"].get("llm_rejected"):
                        override_result = await self.llm_validator.review_rejected_signal(
                            signal=context["strategy_proposal"],
                            rejection_reason=rejection_reason,
                            market_context=context,
                        )
                        context["llm_override_result"] = override_result

                        if override_result.decision == LLMDecision.OVERRIDE_APPROVE:
                            # LLM overrides the rejection!
                            logger.info(
                                f"LLM OVERRIDE APPROVED {market_data.get('symbol', 'BTCUSDT')}: "
                                f"Original rejection: {rejection_reason[:50]}... | "
                                f"Override reason: {override_result.override_reason}"
                            )

                            # Mark signal as LLM override
                            context["strategy_proposal"]["llm_override"] = True
                            context["strategy_proposal"]["llm_override_reason"] = override_result.override_reason

                            # Apply confidence adjustment (ensure minimum confidence)
                            original_confidence = context["strategy_proposal"].get("confidence", 0.5)
                            new_confidence = max(0.55, original_confidence + override_result.confidence_adjustment)
                            context["strategy_proposal"]["confidence"] = new_confidence

                            # Continue to Risk Manager (skip normal Portfolio rejection)
                            portfolio_result["approved"] = True
                            portfolio_result["llm_override"] = True
                            portfolio_result["reasoning"] = f"LLM Override: {override_result.override_reason}"

                # If still not approved after LLM review, return None
                if not portfolio_result.get("approved", False):
                    return None

            # Apply portfolio-adjusted size and TP/SL
            if "adjusted_size" in portfolio_result:
                context["strategy_proposal"]["size"] = portfolio_result["adjusted_size"]
                logger.debug(f"Portfolio adjusted size: {portfolio_result['adjusted_size']}")

            if portfolio_result.get("adjusted_stop_price"):
                context["strategy_proposal"]["stop_price"] = portfolio_result["adjusted_stop_price"]
                logger.debug(f"Portfolio adjusted SL: {portfolio_result['adjusted_stop_price']}")

            if portfolio_result.get("adjusted_target_price"):
                context["strategy_proposal"]["target_price"] = portfolio_result["adjusted_target_price"]
                logger.debug(f"Portfolio adjusted TP: {portfolio_result['adjusted_target_price']}")

        # Stage 4: Risk Assessment (additional checks)
        if "risk_manager" in self.agents:
            risk_result = await self._run_agent("risk_manager", context)
            context["risk_assessment"] = risk_result

            if not risk_result.get("approved", False):
                logger.info(f"Trade rejected by risk manager: {risk_result.get('reason')}")
                return None

            # Apply risk adjustments (only if adjusted_size is not None)
            if risk_result.get("adjusted_size") is not None:
                context["strategy_proposal"]["size"] = risk_result["adjusted_size"]

            logger.debug(f"Risk approved: {risk_result}")

        # Stage 4: Execution Planning
        if "execution_agent" in self.agents:
            execution = await self._run_agent("execution_agent", context)
            context["execution_plan"] = execution
            logger.debug(f"Execution plan: {execution}")

        # Build final signal
        return self._build_signal(context)

    async def _generate_strategy_by_regime(
        self,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Run ALL strategies and pick the best signal.

        Regime is used as a confidence MODIFIER (+20%), not a router.

        Args:
            context: Current context with regime information

        Returns:
            Best strategy result with regime-adjusted confidence
        """
        recommended = context.get("recommended_strategy", "neutral")
        logger.debug(f"Running ALL strategies (regime hint: {recommended})")

        signals = []
        strategy_analysis = []  # Track all strategy analysis for logging

        # Strategy name to regime mapping for confidence boost
        strategy_regime_map = {
            "mean_reversion": "mean_reversion",
            "trend_following": "trend_following",
            "turtle_trading": "turtle_trading",  # Also trend-like
            "momentum": "trend_following",  # Momentum aligns with trend
            "pivot": "mean_reversion",  # Pivot bounces align with mean reversion
            "pattern": "trend_following",  # Breakout patterns align with trend
        }

        # Run Mean Reversion
        if "mean_reversion" in self.agents:
            mr_result = await self._run_agent("mean_reversion", context)
            mr_reasoning = mr_result.get("reasoning", "No analysis")
            strategy_analysis.append(("Mean Reversion", mr_result.get("signal") is not None, mr_reasoning))
            if mr_result.get("signal"):
                mr_result["strategy_source"] = "mean_reversion"
                # Apply regime confidence boost if aligned
                if recommended == strategy_regime_map.get("mean_reversion"):
                    mr_result["confidence"] = min(1.0, mr_result.get("confidence", 0.5) * 1.2)
                    logger.debug("Mean Reversion: +20% regime boost applied")
                signals.append(mr_result)
            else:
                logger.debug(f"Mean Reversion: {mr_reasoning}")

        # Run Trend Following
        if "trend_following" in self.agents:
            tf_result = await self._run_agent("trend_following", context)
            tf_reasoning = tf_result.get("reasoning", "No analysis")
            strategy_analysis.append(("Trend Following", tf_result.get("signal") is not None, tf_reasoning))
            if tf_result.get("signal"):
                tf_result["strategy_source"] = "trend_following"
                if recommended == strategy_regime_map.get("trend_following"):
                    tf_result["confidence"] = min(1.0, tf_result.get("confidence", 0.5) * 1.2)
                    logger.debug("Trend Following: +20% regime boost applied")
                signals.append(tf_result)
            else:
                logger.debug(f"Trend Following: {tf_reasoning}")

        # Run Turtle Trading
        if "turtle_trading" in self.agents:
            turtle_result = await self._run_agent("turtle_trading", context)
            turtle_reasoning = turtle_result.get("reasoning", "No analysis")
            strategy_analysis.append(("Turtle Trading", turtle_result.get("signal") is not None, turtle_reasoning))
            if turtle_result.get("signal"):
                turtle_result["strategy_source"] = "turtle_trading"
                if recommended in ["turtle_trading", "trend_following"]:
                    turtle_result["confidence"] = min(1.0, turtle_result.get("confidence", 0.5) * 1.2)
                    logger.debug("Turtle Trading: +20% regime boost applied")
                signals.append(turtle_result)
            else:
                logger.debug(f"Turtle Trading: {turtle_reasoning}")

        # Run Momentum
        if "momentum" in self.agents:
            momentum_result = await self._run_agent("momentum", context)
            momentum_reasoning = momentum_result.get("reasoning", "No analysis")
            strategy_analysis.append(("Momentum", momentum_result.get("signal") is not None, momentum_reasoning))
            if momentum_result.get("signal"):
                momentum_result["strategy_source"] = "momentum"
                if recommended == "trend_following":
                    momentum_result["confidence"] = min(1.0, momentum_result.get("confidence", 0.5) * 1.2)
                    logger.debug("Momentum: +20% regime boost applied")
                signals.append(momentum_result)
            else:
                logger.debug(f"Momentum: {momentum_reasoning}")

        # Run Pivot
        if "pivot" in self.agents:
            pivot_result = await self._run_agent("pivot", context)
            pivot_reasoning = pivot_result.get("reasoning", "No analysis")
            strategy_analysis.append(("Pivot", pivot_result.get("signal") is not None, pivot_reasoning))
            if pivot_result.get("signal"):
                pivot_result["strategy_source"] = "pivot"
                if recommended == "mean_reversion":
                    pivot_result["confidence"] = min(1.0, pivot_result.get("confidence", 0.5) * 1.2)
                    logger.debug("Pivot: +20% regime boost applied")
                signals.append(pivot_result)
            else:
                logger.debug(f"Pivot: {pivot_reasoning}")

        # Run Pattern
        if "pattern" in self.agents:
            pattern_result = await self._run_agent("pattern", context)
            pattern_reasoning = pattern_result.get("reasoning", "No analysis")
            strategy_analysis.append(("Pattern", pattern_result.get("signal") is not None, pattern_reasoning))
            if pattern_result.get("signal"):
                pattern_result["strategy_source"] = "pattern"
                if recommended == "trend_following":
                    pattern_result["confidence"] = min(1.0, pattern_result.get("confidence", 0.5) * 1.2)
                    logger.debug("Pattern: +20% regime boost applied")
                signals.append(pattern_result)
            else:
                logger.debug(f"Pattern: {pattern_reasoning}")

        # Store analysis for external access (keyed by symbol)
        symbol = context.get("symbol", "UNKNOWN")
        self.last_strategy_analysis[symbol] = strategy_analysis

        # Return strongest signal (by confidence or position size)
        if signals:
            best = max(signals, key=lambda s: (
                s.get("confidence", 0),
                s.get("signal", {}).get("position_size_pct", 0)
            ))
            logger.info(f"Best signal: {best.get('strategy_source')} "
                       f"(confidence: {best.get('confidence', 0):.2f})")
            return best

        logger.debug("No signals from any strategy")
        return None

    def _convert_signal_to_proposal(
        self,
        strategy_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert new-style signal to legacy proposal format.

        Args:
            strategy_result: Result from strategy agent

        Returns:
            Proposal in legacy format for compatibility
        """
        signal = strategy_result.get("signal", {})

        if not signal:
            return {"action": "hold"}

        direction = signal.get("direction", "none")
        if direction == "none":
            return {"action": "hold"}

        # Map direction to action
        action = "buy" if direction == "long" else "sell"

        # Extract ATR/N value if available (from Turtle, Momentum, etc.)
        atr_value = (
            strategy_result.get("n_value") or  # Turtle trading uses n_value
            signal.get("atr") or
            signal.get("n_value") or
            None
        )

        return {
            "action": action,
            "symbol": signal.get("symbol", "BTCUSDT"),
            "size": signal.get("position_size_pct", 0) * 100,  # Convert to units
            "price": signal.get("entry_price"),
            "stop_price": signal.get("stop_price"),
            "target_price": signal.get("target_price") or signal.get("initial_target"),
            "strategy": signal.get("strategy"),
            "timeframe": signal.get("timeframe", "4h"),  # Include timeframe for SL calculation
            "atr": atr_value,  # Include ATR for dynamic stop loss
            "reason": strategy_result.get("reasoning", ""),
            "confidence": strategy_result.get("confidence", 0.5),
            "explanation": strategy_result.get("reasoning", ""),
        }

    async def _run_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single agent and log the decision.

        Args:
            agent_name: Name of agent to run
            context: Current context

        Returns:
            Agent output
        """
        agent = self.agents.get(agent_name)
        if not agent:
            logger.warning(f"Agent not found: {agent_name}")
            return {}

        try:
            # Run agent
            result = await agent.process(context)

            # Log AI decision
            await self.ai_logger.log_decision(
                stage=agent.stage_name,
                model=agent.model_name,
                input_data=agent.get_input_summary(context),
                output_data=result,
                explanation=result.get("explanation", f"{agent_name} decision"),
                agent_name=agent_name,
                confidence=result.get("confidence"),
                duration_ms=result.get("duration_ms"),
                tokens_used=result.get("tokens_used"),
            )

            return result

        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            return {"error": str(e)}

    def _build_signal(self, context: Dict[str, Any]) -> Optional[Signal]:
        """Build a trading signal from the context.

        Args:
            context: Full context with all agent outputs

        Returns:
            Trading signal or None
        """
        proposal = context.get("strategy_proposal", {})
        execution = context.get("execution_plan", {})

        if not proposal:
            return None

        action_str = proposal.get("action", "hold").lower()
        if action_str == "hold":
            return None

        action = SignalAction.BUY if action_str == "buy" else SignalAction.SELL

        return Signal(
            action=action,
            symbol=proposal.get("symbol", "BTCUSDT"),
            size=proposal.get("size", 0),
            price=execution.get("price") or proposal.get("price"),
            stop_price=proposal.get("stop_price"),
            target_price=proposal.get("target_price"),
            reason=proposal.get("reason", "Multi-agent decision"),
            confidence=proposal.get("confidence", 0.5),
            strategy=proposal.get("strategy", "unknown"),
            timeframe=proposal.get("timeframe", "4h"),  # Include timeframe from strategy
            model_used=self._get_primary_model(),
            ai_explanation=self._build_explanation(context),
            metadata={
                "market_analysis": context.get("market_analysis"),
                "risk_assessment": context.get("risk_assessment"),
                "execution_plan": context.get("execution_plan"),
                "regime": context.get("regime", {}),
                "alpha_confirmation": self._serialize_alpha_signal(context.get("alpha_confirmation")),
                "llm_validation": self._serialize_llm_validation(context.get("llm_validation")),
                "llm_override": proposal.get("llm_override", False),
                "llm_override_reason": proposal.get("llm_override_reason"),
            },
        )

    def _get_primary_model(self) -> str:
        """Get the primary AI model name for logging."""
        # Use strategy generator's model as primary
        if "strategy_generator" in self.agents:
            return self.agents["strategy_generator"].model_name
        return "multi-agent-ensemble"

    def _build_explanation(self, context: Dict[str, Any]) -> str:
        """Build combined explanation from all agents.

        Args:
            context: Full context

        Returns:
            Combined explanation string (max 1000 chars)
        """
        parts = []

        if analysis := context.get("market_analysis", {}).get("explanation"):
            parts.append(f"Analysis: {analysis}")

        if strategy := context.get("strategy_proposal", {}).get("explanation"):
            parts.append(f"Strategy: {strategy}")

        if risk := context.get("risk_assessment", {}).get("explanation"):
            parts.append(f"Risk: {risk}")

        explanation = " | ".join(parts)
        return explanation[:1000]  # Max 1000 chars for WEEX

    def _serialize_alpha_signal(self, alpha_signal) -> Optional[Dict[str, Any]]:
        """Serialize AlphaSignal for metadata storage.

        Args:
            alpha_signal: AlphaSignal object or None

        Returns:
            Dictionary representation or None
        """
        if not alpha_signal:
            return None

        return {
            "direction": alpha_signal.direction,
            "alpha": alpha_signal.alpha,
            "confidence": alpha_signal.confidence,
            "components": alpha_signal.components,
            "timeframe_alignment": alpha_signal.timeframe_alignment,
            "active_patterns": alpha_signal.active_patterns[:5],
        }

    def _serialize_llm_validation(self, llm_result) -> Optional[Dict[str, Any]]:
        """Serialize LLMValidationResult for metadata storage.

        Args:
            llm_result: LLMValidationResult object or None

        Returns:
            Dictionary representation or None
        """
        if not llm_result:
            return None

        return {
            "decision": llm_result.decision.value if hasattr(llm_result.decision, 'value') else str(llm_result.decision),
            "confidence_adjustment": llm_result.confidence_adjustment,
            "reasoning": llm_result.reasoning[:500] if llm_result.reasoning else "",
            "market_context": llm_result.market_context[:200] if llm_result.market_context else "",
            "risk_assessment": llm_result.risk_assessment[:200] if llm_result.risk_assessment else "",
            "key_factors": llm_result.key_factors[:5] if llm_result.key_factors else [],
        }

    async def _validate_with_alpha_generator(
        self,
        proposal: Dict[str, Any],
        symbol: str,
    ):
        """Validate strategy signal with AlphaGenerator.

        Uses multi-indicator aggregation to confirm or reject signals.

        Args:
            proposal: Strategy proposal
            symbol: Trading symbol

        Returns:
            AlphaSignal or None
        """
        if not self.alpha_generator:
            return None

        try:
            alpha_signal = await self.alpha_generator.generate_alpha(symbol)

            # Log alpha details
            logger.debug(
                f"Alpha signal: direction={alpha_signal.direction}, "
                f"alpha={alpha_signal.alpha:+.2f}, "
                f"confidence={alpha_signal.confidence:.2f}, "
                f"patterns={alpha_signal.active_patterns[:3]}"
            )

            # Log component breakdown
            logger.debug(f"Alpha components: {alpha_signal.components}")
            logger.debug(f"Timeframe alignment: {alpha_signal.timeframe_alignment}")

            return alpha_signal

        except Exception as e:
            logger.warning(f"AlphaGenerator validation failed: {e}")
            return None

    async def _validate_with_edge_scanner(
        self,
        proposal: Dict[str, Any],
        symbol: str,
    ):
        """Validate strategy signal with EdgeScanner.

        Scans for active edge signals that align with the strategy.

        Args:
            proposal: Strategy proposal
            symbol: Trading symbol

        Returns:
            List of EdgeSignal objects or empty list
        """
        if not self.edge_scanner:
            return []

        try:
            # Scan for edge signals on this symbol
            edge_signals = await self.edge_scanner.scan_symbol_edges(symbol)

            if edge_signals:
                # Log edge details
                for signal in edge_signals[:3]:  # Log first 3
                    logger.debug(
                        f"Edge signal: edge_id={signal.edge_id}, "
                        f"side={signal.side}, "
                        f"price={signal.price:,.2f}, "
                        f"expectancy={signal.edge.expectancy:.2f}" if signal.edge else ""
                    )

            return edge_signals

        except Exception as e:
            logger.warning(f"EdgeScanner validation failed: {e}")
            return []

    async def start(self):
        """Start all agents."""
        self._running = True
        for name, agent in self.agents.items():
            await agent.start()
            logger.info(f"Started agent: {name}")

    async def stop(self):
        """Stop all agents."""
        self._running = False
        for name, agent in self.agents.items():
            await agent.stop()
            logger.info(f"Stopped agent: {name}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "running": self._running,
            "agents": list(self.agents.keys()),
            "agent_count": len(self.agents),
            "ai_logging_stats": self.ai_logger.get_stats(),
        }
