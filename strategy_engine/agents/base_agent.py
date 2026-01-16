"""Base Agent class for multi-agent trading."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for AI trading agents.

    Each agent specializes in one aspect of the trading decision:
    - Market analysis
    - Strategy generation
    - Risk management
    - Execution optimization
    """

    # Override in subclasses
    name: str = "base_agent"
    stage_name: str = "Base Stage"
    model_name: str = "unknown"

    def __init__(self, config: Dict[str, Any]):
        """Initialize agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._running = False

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process context and return decision.

        Args:
            context: Current context with market data and previous agent outputs

        Returns:
            Agent decision with at minimum:
            - explanation: str - Human-readable explanation
            - confidence: float - Confidence score (0-1)
        """
        pass

    def get_input_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get summarized input for AI logging.

        Override to customize what gets logged as input.

        Args:
            context: Full context

        Returns:
            Summarized input for logging
        """
        # Default: return a summary to avoid huge logs
        summary = {}
        if "market_data" in context:
            md = context["market_data"]
            summary["market_data"] = {
                "symbol": md.get("symbol"),
                "price": md.get("price"),
                "timestamp": md.get("timestamp"),
            }
        return summary

    async def start(self):
        """Called when agent starts."""
        self._running = True
        logger.info(f"Agent {self.name} started")

    async def stop(self):
        """Called when agent stops."""
        self._running = False
        logger.info(f"Agent {self.name} stopped")

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running
