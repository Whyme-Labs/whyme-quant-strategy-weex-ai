"""AI Agents for multi-agent trading strategy.

Architecture based on research:
- Regime Detector: Classifies market conditions
- Mean Reversion Agent: Buy low, sell high (fade extremes)
- Trend Following Agent: Buy high, sell higher (ride trends)
- Risk Manager: Position sizing and risk control
- Execution Agent: Order optimization
"""

from .base_agent import BaseAgent
from .market_analyst import MarketAnalystAgent
from .risk_manager import RiskManagerAgent
from .execution import ExecutionAgent
from .regime_detector import RegimeDetectorAgent
from .mean_reversion import MeanReversionAgent
from .trend_following import TrendFollowingAgent
from .portfolio_manager import PortfolioManagerAgent

__all__ = [
    "BaseAgent",
    "MarketAnalystAgent",
    "RiskManagerAgent",
    "ExecutionAgent",
    "RegimeDetectorAgent",
    "MeanReversionAgent",
    "TrendFollowingAgent",
    "PortfolioManagerAgent",
]
