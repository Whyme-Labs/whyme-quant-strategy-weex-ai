"""AI Agents for multi-agent trading strategy.

Architecture:
- Read-Only Analysts:
  - Regime Detector: Classifies market conditions
  - Mean Reversion Agent: Buy low, sell high (fade extremes)
  - Trend Following Agent: Buy high, sell higher (ride trends)
  - Turtle Trading Agent: Classic breakout system (20/55-day channels)
  - Momentum Agent: Trade with price momentum (ROC, RSI, MACD)
  - Pivot Agent: Trade pivot point support/resistance levels
  - Pattern Agent: Trade classical chart patterns (H&S, flags, wedges, triangles)
  - Risk Manager: Position sizing and risk control
  - Portfolio Manager: Gatekeeper, approves/rejects signals

- Executor (Single Point of Execution):
  - Executor Agent: ONLY component that can execute transactions

- Self-Evolving RL Agents:
  - Reflection Agent: Reviews open positions, suggests actions
  - Judge Agent: Scores completed trades on multi-objective reward
  - Learner Agent: Extracts patterns, evolves parameters
"""

from .base_agent import BaseAgent
from .market_analyst import MarketAnalystAgent
from .risk_manager import RiskManagerAgent
from .execution import ExecutorAgent
from .regime_detector import RegimeDetectorAgent
from .mean_reversion import MeanReversionAgent
from .trend_following import TrendFollowingAgent
from .turtle_trading import TurtleTradingAgent
from .momentum import MomentumAgent
from .pivot import PivotAgent
from .pattern_agent import PatternAgent
from .portfolio_manager import PortfolioManagerAgent

# Self-evolving RL agents
from .reflection_agent import ReflectionAgent
from .judge_agent import JudgeAgent
from .learner_agent import LearnerAgent

__all__ = [
    # Base
    "BaseAgent",
    # Read-only analyst agents
    "MarketAnalystAgent",
    "RiskManagerAgent",
    "RegimeDetectorAgent",
    "MeanReversionAgent",
    "TrendFollowingAgent",
    "TurtleTradingAgent",
    "MomentumAgent",
    "PivotAgent",
    "PatternAgent",
    "PortfolioManagerAgent",
    # Executor (single point of execution)
    "ExecutorAgent",
    # Self-evolving RL agents
    "ReflectionAgent",
    "JudgeAgent",
    "LearnerAgent",
]
