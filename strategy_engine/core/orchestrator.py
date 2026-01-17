"""Agent Orchestrator for multi-agent trading.

Coordinates multiple AI agents to make collaborative trading decisions.

Architecture (based on research insights):
- Regime Detector determines whether to use Mean Reversion or Trend Following
- "Some regimes reward trend following. Others reward mean reversion."
- Running both strategies across different regimes smooths returns and reduces drawdowns.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from .base import Signal, SignalAction
from ..agents.base_agent import BaseAgent
from ai_logging import get_ai_logger


class AgentOrchestrator:
    """Orchestrates multiple AI agents for trading decisions.

    The orchestrator manages the flow of information between agents:
    1. Regime Detector classifies market conditions
    2. Routes to appropriate strategy:
       - Mean Reversion (high win rate, fade extremes)
       - Trend Following (low win rate, big winners)
    3. Risk Manager evaluates and adjusts the proposal
    4. Execution Agent optimizes order execution
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.ai_logger = get_ai_logger()
        self._running = False
        self.last_regime = None

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
        context = {"market_data": market_data, "timestamp": market_data.get("timestamp")}

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

        # Stage 3: Portfolio Management (dynamic confidence threshold)
        # This is the bridge between signals and execution
        if "portfolio_manager" in self.agents:
            portfolio_result = await self._run_agent("portfolio_manager", context)
            context["portfolio_decision"] = portfolio_result

            if not portfolio_result.get("approved", False):
                logger.info(
                    f"Trade rejected by portfolio manager: {portfolio_result.get('reasoning')}. "
                    f"Signal confidence: {portfolio_result.get('signal_confidence', 0):.2f}, "
                    f"Threshold: {portfolio_result.get('confidence_threshold', 0):.2f}"
                )
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

            # Apply risk adjustments
            if "adjusted_size" in risk_result:
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
        """Route to appropriate strategy agent based on detected regime.

        Args:
            context: Current context with regime information

        Returns:
            Strategy result from the appropriate agent
        """
        recommended = context.get("recommended_strategy", "neutral")

        # Route based on regime recommendation
        if recommended == "mean_reversion" and "mean_reversion" in self.agents:
            logger.debug("Routing to Mean Reversion strategy")
            return await self._run_agent("mean_reversion", context)

        elif recommended == "trend_following" and "trend_following" in self.agents:
            logger.debug("Routing to Trend Following strategy")
            return await self._run_agent("trend_following", context)

        elif recommended == "neutral":
            # In neutral regime, we can either:
            # 1. Skip trading (conservative)
            # 2. Check both strategies and take the stronger signal
            logger.debug("Neutral regime - checking both strategies")

            signals = []

            if "mean_reversion" in self.agents:
                mr_result = await self._run_agent("mean_reversion", context)
                if mr_result.get("signal"):
                    mr_result["strategy_source"] = "mean_reversion"
                    signals.append(mr_result)

            if "trend_following" in self.agents:
                tf_result = await self._run_agent("trend_following", context)
                if tf_result.get("signal"):
                    tf_result["strategy_source"] = "trend_following"
                    signals.append(tf_result)

            # Return strongest signal (by position size as proxy for confidence)
            if signals:
                best = max(signals, key=lambda s: s.get("signal", {}).get("position_size_pct", 0))
                logger.debug(f"Best signal in neutral regime: {best.get('strategy_source')}")
                return best

            return None

        # Fallback to legacy strategy_generator if no new agents
        elif "strategy_generator" in self.agents:
            return await self._run_agent("strategy_generator", context)

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

        return {
            "action": action,
            "symbol": signal.get("symbol", "BTCUSDT"),
            "size": signal.get("position_size_pct", 0) * 100,  # Convert to units
            "price": signal.get("entry_price"),
            "stop_price": signal.get("stop_price"),
            "target_price": signal.get("target_price") or signal.get("initial_target"),
            "strategy": signal.get("strategy"),
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
            model_used=self._get_primary_model(),
            ai_explanation=self._build_explanation(context),
            metadata={
                "market_analysis": context.get("market_analysis"),
                "risk_assessment": context.get("risk_assessment"),
                "execution_plan": context.get("execution_plan"),
                "regime": context.get("regime", {}),
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
