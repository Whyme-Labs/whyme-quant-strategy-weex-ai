"""AI Decision Logger for WEEX Hackathon.

Captures all AI decisions for mandatory hackathon verification.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .models import AIDecision, TradeDecision, AILogBatch


class AILogger:
    """Centralized AI decision logger.

    Captures all AI model inputs, outputs, and reasoning for hackathon compliance.
    """

    # Standard stage names for the hackathon
    STAGE_MARKET_ANALYSIS = "Market Analysis"
    STAGE_STRATEGY_GENERATION = "Strategy Generation"
    STAGE_RISK_ASSESSMENT = "Risk Assessment"
    STAGE_EXECUTION_PLANNING = "Execution Planning"
    STAGE_ORDER_PLACEMENT = "Order Placement"

    def __init__(self):
        """Initialize the AI logger."""
        self._decisions: List[AIDecision] = []
        self._pending_uploads: List[AIDecision] = []
        self._trade_decisions: Dict[str, TradeDecision] = {}
        self._lock = asyncio.Lock()

    async def log_decision(
        self,
        stage: str,
        model: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        explanation: str,
        order_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
    ) -> AIDecision:
        """Log an AI decision.

        Args:
            stage: Trading stage name
            model: AI model identifier
            input_data: Input provided to the AI
            output_data: Output from the AI
            explanation: Natural language explanation (max 1000 chars)
            order_id: Optional WEEX order ID
            agent_name: Name of the agent making the decision
            confidence: Confidence score (0-1)
            duration_ms: Inference time in milliseconds
            tokens_used: Number of tokens consumed

        Returns:
            The created AIDecision record
        """
        decision = AIDecision(
            stage=stage,
            model=model,
            input_data=input_data,
            output_data=output_data,
            explanation=explanation[:1000],
            order_id=order_id,
            agent_name=agent_name,
            confidence=confidence,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
        )

        async with self._lock:
            self._decisions.append(decision)
            self._pending_uploads.append(decision)

        logger.info(
            f"AI Decision logged: stage={stage}, model={model}, "
            f"agent={agent_name}, confidence={confidence}"
        )

        return decision

    async def log_market_analysis(
        self,
        model: str,
        market_data: Dict[str, Any],
        analysis_result: Dict[str, Any],
        explanation: str,
        agent_name: str = "MarketAnalyst",
        **kwargs,
    ) -> AIDecision:
        """Log a market analysis decision.

        Args:
            model: AI model used
            market_data: Input market data
            analysis_result: Analysis output
            explanation: Reasoning explanation
            agent_name: Agent name
            **kwargs: Additional parameters

        Returns:
            AIDecision record
        """
        return await self.log_decision(
            stage=self.STAGE_MARKET_ANALYSIS,
            model=model,
            input_data={"market_data": market_data},
            output_data=analysis_result,
            explanation=explanation,
            agent_name=agent_name,
            **kwargs,
        )

    async def log_strategy_generation(
        self,
        model: str,
        context: Dict[str, Any],
        strategy_output: Dict[str, Any],
        explanation: str,
        agent_name: str = "StrategyGenerator",
        **kwargs,
    ) -> AIDecision:
        """Log a strategy generation decision.

        Args:
            model: AI model used
            context: Input context (market analysis, etc.)
            strategy_output: Generated strategy
            explanation: Reasoning explanation
            agent_name: Agent name
            **kwargs: Additional parameters

        Returns:
            AIDecision record
        """
        return await self.log_decision(
            stage=self.STAGE_STRATEGY_GENERATION,
            model=model,
            input_data=context,
            output_data=strategy_output,
            explanation=explanation,
            agent_name=agent_name,
            **kwargs,
        )

    async def log_risk_assessment(
        self,
        model: str,
        position_data: Dict[str, Any],
        risk_output: Dict[str, Any],
        explanation: str,
        agent_name: str = "RiskManager",
        **kwargs,
    ) -> AIDecision:
        """Log a risk assessment decision.

        Args:
            model: AI model used
            position_data: Current position and proposed trade
            risk_output: Risk assessment output
            explanation: Reasoning explanation
            agent_name: Agent name
            **kwargs: Additional parameters

        Returns:
            AIDecision record
        """
        return await self.log_decision(
            stage=self.STAGE_RISK_ASSESSMENT,
            model=model,
            input_data=position_data,
            output_data=risk_output,
            explanation=explanation,
            agent_name=agent_name,
            **kwargs,
        )

    async def log_execution(
        self,
        model: str,
        order_params: Dict[str, Any],
        execution_plan: Dict[str, Any],
        explanation: str,
        order_id: Optional[int] = None,
        agent_name: str = "ExecutionAgent",
        **kwargs,
    ) -> AIDecision:
        """Log an execution decision.

        Args:
            model: AI model used
            order_params: Order parameters
            execution_plan: Execution output
            explanation: Reasoning explanation
            order_id: WEEX order ID
            agent_name: Agent name
            **kwargs: Additional parameters

        Returns:
            AIDecision record
        """
        return await self.log_decision(
            stage=self.STAGE_EXECUTION_PLANNING,
            model=model,
            input_data=order_params,
            output_data=execution_plan,
            explanation=explanation,
            order_id=order_id,
            agent_name=agent_name,
            **kwargs,
        )

    async def get_pending_uploads(self) -> List[AIDecision]:
        """Get decisions pending upload.

        Returns:
            List of pending AIDecision records
        """
        async with self._lock:
            pending = self._pending_uploads.copy()
            return pending

    async def mark_uploaded(self, decisions: List[AIDecision]):
        """Mark decisions as uploaded.

        Args:
            decisions: Decisions that were uploaded
        """
        async with self._lock:
            for decision in decisions:
                if decision in self._pending_uploads:
                    self._pending_uploads.remove(decision)

    async def get_all_decisions(self) -> List[AIDecision]:
        """Get all logged decisions.

        Returns:
            List of all AIDecision records
        """
        async with self._lock:
            return self._decisions.copy()

    async def get_decisions_for_order(self, order_id: int) -> List[AIDecision]:
        """Get all decisions associated with an order.

        Args:
            order_id: WEEX order ID

        Returns:
            List of AIDecision records for the order
        """
        async with self._lock:
            return [d for d in self._decisions if d.order_id == order_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.

        Returns:
            Dictionary with logging stats
        """
        return {
            "total_decisions": len(self._decisions),
            "pending_uploads": len(self._pending_uploads),
            "decisions_by_stage": self._count_by_stage(),
            "decisions_by_model": self._count_by_model(),
        }

    def _count_by_stage(self) -> Dict[str, int]:
        """Count decisions by stage."""
        counts: Dict[str, int] = {}
        for decision in self._decisions:
            counts[decision.stage] = counts.get(decision.stage, 0) + 1
        return counts

    def _count_by_model(self) -> Dict[str, int]:
        """Count decisions by model."""
        counts: Dict[str, int] = {}
        for decision in self._decisions:
            counts[decision.model] = counts.get(decision.model, 0) + 1
        return counts


# Global singleton instance
_ai_logger: Optional[AILogger] = None


def get_ai_logger() -> AILogger:
    """Get the global AI logger instance.

    Returns:
        AILogger singleton
    """
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = AILogger()
    return _ai_logger
