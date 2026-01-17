"""AI Agents for multi-agent trading strategy.

Architecture based on research:
- Regime Detector: Classifies market conditions
- Mean Reversion Agent: Buy low, sell high (fade extremes)
- Trend Following Agent: Buy high, sell higher (ride trends)
- Turtle Trading Agent: Classic breakout system (20/55-day channels)
- Risk Manager: Position sizing and risk control
- Execution Agent: Order optimization

Self-Evolving RL Agents:
- Reflection Agent: Reviews open positions, suggests actions
- Judge Agent: Scores completed trades on multi-objective reward
- Learner Agent: Extracts patterns, evolves parameters
"""

from .base_agent import BaseAgent
from .market_analyst import MarketAnalystAgent
from .risk_manager import RiskManagerAgent
from .execution import ExecutionAgent
from .regime_detector import RegimeDetectorAgent
from .mean_reversion import MeanReversionAgent
from .trend_following import TrendFollowingAgent
from .turtle_trading import TurtleTradingAgent
from .portfolio_manager import PortfolioManagerAgent

# Self-evolving RL agents
from .reflection_agent import ReflectionAgent
from .judge_agent import JudgeAgent
from .learner_agent import LearnerAgent

__all__ = [
    # Base
    "BaseAgent",
    # Trading agents
    "MarketAnalystAgent",
    "RiskManagerAgent",
    "ExecutionAgent",
    "RegimeDetectorAgent",
    "MeanReversionAgent",
    "TrendFollowingAgent",
    "TurtleTradingAgent",
    "PortfolioManagerAgent",
    # Self-evolving RL agents
    "ReflectionAgent",
    "JudgeAgent",
    "LearnerAgent",
]
