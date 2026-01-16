"""AI Logging System for WEEX Hackathon.

This module handles all AI decision logging required for hackathon verification.
"""

from .logger import AILogger, get_ai_logger
from .models import AIDecision
from .uploader import AILogUploader

# Stage constants for convenience
STAGE_MARKET_ANALYSIS = AILogger.STAGE_MARKET_ANALYSIS
STAGE_STRATEGY_GENERATION = AILogger.STAGE_STRATEGY_GENERATION
STAGE_RISK_ASSESSMENT = AILogger.STAGE_RISK_ASSESSMENT
STAGE_EXECUTION_PLANNING = AILogger.STAGE_EXECUTION_PLANNING
STAGE_ORDER_PLACEMENT = AILogger.STAGE_ORDER_PLACEMENT

__all__ = [
    "AILogger",
    "AIDecision",
    "AILogUploader",
    "get_ai_logger",
    "STAGE_MARKET_ANALYSIS",
    "STAGE_STRATEGY_GENERATION",
    "STAGE_RISK_ASSESSMENT",
    "STAGE_EXECUTION_PLANNING",
    "STAGE_ORDER_PLACEMENT",
]
