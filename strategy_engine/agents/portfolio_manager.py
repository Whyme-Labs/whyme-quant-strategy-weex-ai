"""Portfolio Manager Agent.

The critical bridge between strategy signals and execution.
Determines whether to execute trades based on:
1. Current portfolio positions and exposure
2. Market conditions and regime
3. Dynamic confidence thresholds
4. Correlation with existing positions
5. Risk budget allocation
6. Signal deduplication (prevents same signal firing repeatedly)

"A signal is just a suggestion. Portfolio context determines execution."
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import numpy as np
from loguru import logger

from .base_agent import BaseAgent
from ai_logging import get_ai_logger, STAGE_RISK_ASSESSMENT


class ExecutionDecision(Enum):
    """Final execution decision."""
    EXECUTE = "execute"
    REDUCE_SIZE = "reduce_size"
    REJECT = "reject"
    QUEUE = "queue"  # Wait for better conditions


@dataclass
class DynamicStopLossResult:
    """Result of dynamic stop loss calculation."""
    stop_price: float
    target_price: float
    risk_reward_ratio: float
    stop_distance_pct: float
    target_distance_pct: float
    atr_multiplier_used: float
    confidence_adjustment: float
    regime_adjustment: float
    key_level_adjustment: float
    reasoning: str


class DynamicStopLossCalculator:
    """Dynamic Stop Loss Calculator.

    Calculates optimal stop loss and take profit levels based on:
    1. ATR (Average True Range) for volatility-adaptive stops
    2. Signal confidence for risk scaling
    3. Target risk-reward ratios that scale with confidence
    4. Market regime adjustments
    5. Key support/resistance level awareness

    Philosophy:
    - Higher confidence = tighter stops allowed (smaller risk per trade)
    - Lower confidence = wider stops required (more room for noise)
    - R:R ratio scales with confidence (high conf = target higher R:R)
    - Never place stops at obvious levels (add buffer beyond S/R)
    """

    # Base ATR multipliers for stop distance
    BASE_ATR_MULTIPLIER = 2.0

    # Confidence-based stop adjustment factors
    # Higher confidence = can use tighter stop (smaller multiplier)
    CONFIDENCE_STOP_ADJUSTMENTS = {
        "very_high": 0.75,   # Confidence > 0.85: tighter stop
        "high": 0.85,        # Confidence 0.75-0.85: slightly tighter
        "medium": 1.0,       # Confidence 0.60-0.75: standard
        "low": 1.2,          # Confidence 0.45-0.60: wider stop
        "very_low": 1.4,     # Confidence < 0.45: much wider
    }

    # Confidence-based R:R targets
    # Higher confidence = can target higher R:R
    CONFIDENCE_RR_TARGETS = {
        "very_high": 2.5,    # Confidence > 0.85: aim for 2.5:1
        "high": 2.0,         # Confidence 0.75-0.85: aim for 2:1
        "medium": 1.75,      # Confidence 0.60-0.75: aim for 1.75:1
        "low": 1.5,          # Confidence 0.45-0.60: minimum 1.5:1
        "very_low": 1.25,    # Confidence < 0.45: may reduce to 1.25:1
    }

    # Regime-based adjustments
    REGIME_ADJUSTMENTS = {
        "high": 1.3,         # High volatility: widen stop by 30%
        "medium": 1.0,       # Normal volatility: no change
        "low": 0.9,          # Low volatility: tighten stop by 10%
    }

    # Key level buffer (how far beyond S/R to place stop)
    KEY_LEVEL_BUFFER_PCT = 0.003  # 0.3% beyond key level

    def __init__(
        self,
        base_atr_multiplier: float = 2.0,
        min_rr_ratio: float = 1.25,
        max_stop_distance_pct: float = 0.05,  # Max 5% stop distance
        min_stop_distance_pct: float = 0.005,  # Min 0.5% stop distance
    ):
        """Initialize dynamic stop loss calculator.

        Args:
            base_atr_multiplier: Base ATR multiplier for stop distance
            min_rr_ratio: Minimum acceptable risk-reward ratio
            max_stop_distance_pct: Maximum stop distance as % of entry
            min_stop_distance_pct: Minimum stop distance as % of entry
        """
        self.base_atr_multiplier = base_atr_multiplier
        self.min_rr_ratio = min_rr_ratio
        self.max_stop_distance_pct = max_stop_distance_pct
        self.min_stop_distance_pct = min_stop_distance_pct

    def calculate(
        self,
        entry_price: float,
        direction: str,  # "long" or "short"
        confidence: float,
        atr: Optional[float] = None,
        regime_volatility: str = "medium",
        key_levels: Optional[Dict[str, Any]] = None,
        original_stop: Optional[float] = None,
        original_target: Optional[float] = None,
    ) -> DynamicStopLossResult:
        """Calculate dynamic stop loss and take profit.

        Args:
            entry_price: Entry price for the trade
            direction: Trade direction ("long" or "short")
            confidence: Signal confidence (0.0 - 1.0)
            atr: Average True Range value (optional, will estimate if not provided)
            regime_volatility: Market volatility regime ("high", "medium", "low")
            key_levels: Dict with "supports" and "resistances" lists
            original_stop: Original stop price from strategy (for reference)
            original_target: Original target price from strategy (for reference)

        Returns:
            DynamicStopLossResult with calculated levels
        """
        reasoning_parts = []

        # 1. Determine confidence tier
        confidence_tier = self._get_confidence_tier(confidence)
        reasoning_parts.append(f"Confidence tier: {confidence_tier} ({confidence:.2f})")

        # 2. Get confidence-based adjustments
        confidence_stop_adj = self.CONFIDENCE_STOP_ADJUSTMENTS[confidence_tier]
        target_rr = self.CONFIDENCE_RR_TARGETS[confidence_tier]
        reasoning_parts.append(f"Target R:R: {target_rr}:1")

        # 3. Get regime-based adjustment
        regime_adj = self.REGIME_ADJUSTMENTS.get(regime_volatility, 1.0)
        if regime_volatility == "high":
            reasoning_parts.append(f"High volatility: +30% stop buffer")
        elif regime_volatility == "low":
            reasoning_parts.append(f"Low volatility: -10% stop buffer")

        # 4. Calculate base stop distance
        if atr and atr > 0:
            # Use ATR-based calculation
            atr_mult_used = self.base_atr_multiplier * confidence_stop_adj * regime_adj
            stop_distance = atr * atr_mult_used
            reasoning_parts.append(f"ATR-based stop: {atr_mult_used:.2f}x ATR")
        elif original_stop and original_stop > 0:
            # Use original stop as reference
            stop_distance = abs(entry_price - original_stop)
            stop_distance *= confidence_stop_adj * regime_adj
            atr_mult_used = 0  # Not ATR-based
            reasoning_parts.append(f"Strategy-based stop adjusted by {confidence_stop_adj * regime_adj:.2f}x")
        else:
            # Fallback: use percentage of price
            base_pct = 0.02  # 2% default
            stop_distance = entry_price * base_pct * confidence_stop_adj * regime_adj
            atr_mult_used = 0
            reasoning_parts.append(f"Default 2% stop adjusted to {base_pct * confidence_stop_adj * regime_adj:.1%}")

        # 5. Enforce min/max stop distance
        stop_distance_pct = stop_distance / entry_price
        if stop_distance_pct > self.max_stop_distance_pct:
            stop_distance = entry_price * self.max_stop_distance_pct
            reasoning_parts.append(f"Capped stop at {self.max_stop_distance_pct:.1%} max")
        elif stop_distance_pct < self.min_stop_distance_pct:
            stop_distance = entry_price * self.min_stop_distance_pct
            reasoning_parts.append(f"Raised stop to {self.min_stop_distance_pct:.1%} min")

        # 6. Calculate initial stop and target prices
        if direction == "long":
            stop_price = entry_price - stop_distance
            target_distance = stop_distance * target_rr
            target_price = entry_price + target_distance
        else:  # short
            stop_price = entry_price + stop_distance
            target_distance = stop_distance * target_rr
            target_price = entry_price - target_distance

        # 7. Key level adjustment
        key_level_adj = 0.0
        if key_levels:
            stop_price, key_level_adj, level_reason = self._adjust_for_key_levels(
                stop_price=stop_price,
                entry_price=entry_price,
                direction=direction,
                key_levels=key_levels,
            )
            if level_reason:
                reasoning_parts.append(level_reason)
                # Recalculate target to maintain R:R
                stop_distance = abs(entry_price - stop_price)
                target_distance = stop_distance * target_rr
                if direction == "long":
                    target_price = entry_price + target_distance
                else:
                    target_price = entry_price - target_distance

        # 8. Final calculations
        final_stop_distance_pct = abs(entry_price - stop_price) / entry_price
        final_target_distance_pct = abs(target_price - entry_price) / entry_price
        final_rr = final_target_distance_pct / final_stop_distance_pct if final_stop_distance_pct > 0 else 0

        # 9. Ensure minimum R:R
        if final_rr < self.min_rr_ratio:
            # Adjust target to meet minimum R:R
            target_distance = stop_distance * self.min_rr_ratio
            if direction == "long":
                target_price = entry_price + target_distance
            else:
                target_price = entry_price - target_distance
            final_rr = self.min_rr_ratio
            reasoning_parts.append(f"Adjusted target to meet min R:R {self.min_rr_ratio}:1")

        return DynamicStopLossResult(
            stop_price=stop_price,
            target_price=target_price,
            risk_reward_ratio=final_rr,
            stop_distance_pct=final_stop_distance_pct,
            target_distance_pct=abs(target_price - entry_price) / entry_price,
            atr_multiplier_used=atr_mult_used,
            confidence_adjustment=confidence_stop_adj,
            regime_adjustment=regime_adj,
            key_level_adjustment=key_level_adj,
            reasoning=" | ".join(reasoning_parts),
        )

    def _get_confidence_tier(self, confidence: float) -> str:
        """Map confidence value to tier."""
        if confidence >= 0.85:
            return "very_high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.60:
            return "medium"
        elif confidence >= 0.45:
            return "low"
        else:
            return "very_low"

    def _adjust_for_key_levels(
        self,
        stop_price: float,
        entry_price: float,
        direction: str,
        key_levels: Dict[str, Any],
    ) -> tuple[float, float, Optional[str]]:
        """Adjust stop price to avoid placing it at obvious key levels.

        Args:
            stop_price: Calculated stop price
            entry_price: Entry price
            direction: Trade direction
            key_levels: Dict with "supports", "resistances", "nearest_support", "nearest_resistance"

        Returns:
            Tuple of (adjusted_stop, adjustment_amount, reason_string)
        """
        supports = key_levels.get("supports", [])
        resistances = key_levels.get("resistances", [])
        nearest_support = key_levels.get("nearest_support")
        nearest_resistance = key_levels.get("nearest_resistance")

        # Combine all levels
        all_levels = []
        if supports:
            all_levels.extend([s.get("price", 0) if isinstance(s, dict) else s for s in supports])
        if resistances:
            all_levels.extend([r.get("price", 0) if isinstance(r, dict) else r for r in resistances])
        if nearest_support:
            level = nearest_support.get("price", 0) if isinstance(nearest_support, dict) else nearest_support
            if level > 0:
                all_levels.append(level)
        if nearest_resistance:
            level = nearest_resistance.get("price", 0) if isinstance(nearest_resistance, dict) else nearest_resistance
            if level > 0:
                all_levels.append(level)

        if not all_levels:
            return stop_price, 0.0, None

        # Check if stop is too close to any key level
        buffer = entry_price * self.KEY_LEVEL_BUFFER_PCT

        for level in all_levels:
            if level <= 0:
                continue

            distance_to_level = abs(stop_price - level)

            # If stop is within buffer distance of a key level
            if distance_to_level < buffer * 2:
                # Adjust stop to be beyond the key level
                if direction == "long":
                    # For longs, stop is below entry - push it below the support level
                    if level < entry_price:  # This is a support level
                        new_stop = level - buffer
                        adjustment = stop_price - new_stop
                        return new_stop, adjustment, f"Stop moved {adjustment:.2f} below support at {level:.2f}"
                else:  # short
                    # For shorts, stop is above entry - push it above the resistance level
                    if level > entry_price:  # This is a resistance level
                        new_stop = level + buffer
                        adjustment = new_stop - stop_price
                        return new_stop, adjustment, f"Stop moved {adjustment:.2f} above resistance at {level:.2f}"

        return stop_price, 0.0, None


@dataclass
class Position:
    """Current portfolio position."""
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


@dataclass
class PortfolioState:
    """Current portfolio state."""
    positions: List[Position] = field(default_factory=list)
    total_equity: float = 0.0
    used_margin: float = 0.0
    available_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    win_rate_recent: float = 0.5  # Last N trades
    max_drawdown_today: float = 0.0


@dataclass
class ExecutionPlan:
    """Final execution plan after portfolio analysis."""
    decision: ExecutionDecision
    original_signal: Dict[str, Any]
    adjusted_size: float
    adjusted_stop_price: Optional[float]
    adjusted_target_price: Optional[float]
    confidence_threshold_used: float
    signal_confidence: float
    reasoning: str
    portfolio_impact: Dict[str, Any]


@dataclass
class SignalSignature:
    """Signature of a trading signal for deduplication."""
    signature_hash: str
    symbol: str
    strategy: str
    direction: str
    timeframe: str
    executed_at: datetime
    reasoning_summary: str


class PortfolioManagerAgent(BaseAgent):
    """Portfolio Manager - The execution gatekeeper.

    This agent sits between strategy signals and actual execution.
    It makes the final call on whether to trade based on portfolio context.

    Key Responsibilities:
    1. Track all current positions
    2. Calculate portfolio-level metrics (exposure, correlation, drawdown)
    3. Dynamically adjust confidence thresholds
    4. Approve/reject/modify trade proposals
    5. Manage risk budget allocation across strategies

    Philosophy:
    - A good signal in a bad portfolio context should be rejected
    - Confidence thresholds should adapt to market conditions
    - Correlation matters - don't pile into correlated positions
    - Preserve capital first, maximize returns second
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize portfolio manager.

        Args:
            config: Configuration with:
                - max_portfolio_exposure: Max total exposure (default 0.5 = 50%)
                - max_single_position: Max single position size (default 0.1 = 10%)
                - max_correlated_exposure: Max exposure to correlated assets (default 0.3)
                - base_confidence_threshold: Starting confidence threshold (default 0.6)
                - max_daily_drawdown: Max daily drawdown before stopping (default 0.05)
                - win_rate_lookback: Number of trades to calculate win rate (default 20)
        """
        super().__init__(config)
        self.max_portfolio_exposure = config.get("max_portfolio_exposure", 0.5)
        self.max_single_position = config.get("max_single_position", 0.1)
        self.max_correlated_exposure = config.get("max_correlated_exposure", 0.3)
        self.base_confidence_threshold = config.get("base_confidence_threshold", 0.6)
        self.max_daily_drawdown = config.get("max_daily_drawdown", 0.05)
        self.win_rate_lookback = config.get("win_rate_lookback", 20)

        # Portfolio state
        self.portfolio = PortfolioState()
        self.trade_history: List[Dict[str, Any]] = []

        # Dynamic Stop Loss Calculator
        self.stop_loss_calculator = DynamicStopLossCalculator(
            base_atr_multiplier=config.get("base_atr_multiplier", 2.0),
            min_rr_ratio=config.get("min_rr_ratio", 1.25),
            max_stop_distance_pct=config.get("max_stop_distance_pct", 0.05),
            min_stop_distance_pct=config.get("min_stop_distance_pct", 0.005),
        )

        # Correlation groups (simplified)
        self.correlation_groups = {
            "btc_related": ["BTCUSDT", "BTCUSD"],
            "eth_related": ["ETHUSDT", "ETHUSD"],
            "major_alts": ["SOLUSDT", "AVAXUSDT", "DOTUSDT"],
        }

        # Signal deduplication tracking
        # Prevents same signal from firing repeatedly
        self._executed_signals: Dict[str, SignalSignature] = {}  # hash -> signature
        self._signal_cooldowns = {
            "1h": timedelta(hours=2),    # 1H signals: 2 hour cooldown
            "4h": timedelta(hours=8),    # 4H signals: 8 hour cooldown
            "1d": timedelta(hours=24),   # 1D signals: 24 hour cooldown
            "default": timedelta(hours=4),
        }

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trade signal through portfolio lens.

        Args:
            context: Dictionary containing:
                - strategy_proposal: Signal from strategy agent
                - market_data: Current market data
                - regime: Market regime
                - account_info: Optional account information

        Returns:
            Execution plan with final decision
        """
        proposal = context.get("strategy_proposal", {})
        market_data = context.get("market_data", {})
        regime = context.get("regime", {})
        account_info = context.get("account_info", {})

        # ============================================================
        # LLM DECISION CHECKS (highest priority)
        # LLM has FULL override authority over all decisions
        # ============================================================

        # Check 0.1: LLM Rejection (LLM rejected the signal before Portfolio Manager)
        if proposal and proposal.get("llm_rejected"):
            llm_rejection_reason = proposal.get("llm_rejection_reason", "LLM rejected this signal")
            logger.info(f"PORTFOLIO: Respecting LLM rejection: {llm_rejection_reason[:100]}")
            return {
                "decision": ExecutionDecision.REJECT.value,
                "approved": False,
                "reasoning": f"LLM rejected: {llm_rejection_reason}",
                "llm_decision": True,
            }

        # Check 0.2: LLM Override (LLM approved a previously rejected signal)
        if proposal and proposal.get("llm_override"):
            llm_override_reason = proposal.get("llm_override_reason", "LLM approved this signal")
            llm_confidence = proposal.get("confidence", 0.55)
            logger.info(f"PORTFOLIO: Respecting LLM override approval: {llm_override_reason[:100]}")
            # Continue with processing but mark as LLM-approved
            # The signal will still go through margin/exposure checks but not confidence threshold

        # Update portfolio state if account info provided
        if account_info:
            self._update_portfolio_state(account_info)

        # Get account state
        equity = float(account_info.get("equity", 0))
        available_margin = float(account_info.get("available", 0))
        used_margin = float(account_info.get("usedMargin", 0))
        existing_positions = account_info.get("positions", [])

        # Calculate current exposure
        total_position_margin = sum(
            float(p.get("margin", 0)) for p in existing_positions
        )
        current_exposure_pct = (total_position_margin / equity * 100) if equity > 0 else 0

        # ============================================================
        # CRITICAL: Margin and Exposure Checks
        # These are the most important guards against over-trading
        # ============================================================

        # Check 1: Minimum available margin
        min_margin_required = 10  # Minimum $10 available to trade
        if available_margin < min_margin_required:
            return {
                "decision": ExecutionDecision.REJECT.value,
                "approved": False,
                "reasoning": f"Insufficient margin: ${available_margin:.2f} available, minimum ${min_margin_required} required",
            }

        # Check 2: Maximum total exposure limit (50% of equity)
        max_total_exposure_pct = self.max_portfolio_exposure * 100  # Default 50%
        if current_exposure_pct >= max_total_exposure_pct:
            return {
                "decision": ExecutionDecision.REJECT.value,
                "approved": False,
                "reasoning": f"Max portfolio exposure reached: {current_exposure_pct:.1f}% (limit: {max_total_exposure_pct:.0f}%). Cannot open new positions.",
            }

        # Check 3: Ensure we keep reserve margin (at least 30% of equity)
        min_reserve_pct = 30
        min_reserve = equity * (min_reserve_pct / 100)
        if available_margin < min_reserve:
            return {
                "decision": ExecutionDecision.REJECT.value,
                "approved": False,
                "reasoning": f"Must maintain {min_reserve_pct}% reserve. Available: ${available_margin:.2f}, Required reserve: ${min_reserve:.2f}",
            }

        # Check 4: Position and timeframe conflict check
        # Rules:
        # - ONE position per symbol (no mixing strategies/timeframes)
        # - Same direction: Cannot add to existing position
        # - Opposite direction: Cannot open conflicting position
        # This prevents timeframe conflicts (e.g., 4H LONG vs 1D SHORT)
        symbol = proposal.get("symbol", "BTCUSDT") if proposal else None
        proposed_strategy = proposal.get("strategy", "unknown") if proposal else "unknown"
        proposed_timeframe = proposal.get("timeframe", "4h") if proposal else "4h"
        action = proposal.get("action", "").lower() if proposal else ""
        proposed_side = "long" if action == "buy" else "short" if action == "sell" else ""

        if symbol:
            for pos in existing_positions:
                pos_symbol = pos.get("symbol", "").replace("cmt_", "").upper()
                pos_side = pos.get("side", "").lower()
                pos_size = float(pos.get("size", 0))

                # If ANY position exists on this symbol, apply strict rules
                if pos_symbol == symbol and pos_size > 0:
                    if pos_side == proposed_side:
                        # Same direction - don't add to position (prevents repeated trades)
                        return {
                            "decision": ExecutionDecision.REJECT.value,
                            "approved": False,
                            "reasoning": f"Already have open {pos_side.upper()} position on {symbol} (size: {pos_size:.4f}). "
                                        f"Cannot add from {proposed_strategy} ({proposed_timeframe}). One position per symbol.",
                        }
                    else:
                        # Opposite direction - conflicting signal from potentially different timeframe
                        return {
                            "decision": ExecutionDecision.REJECT.value,
                            "approved": False,
                            "reasoning": f"TIMEFRAME CONFLICT: Have {pos_side.upper()} on {symbol}, "
                                        f"but {proposed_strategy} ({proposed_timeframe}) wants {proposed_side.upper()}. "
                                        f"Close existing position first. No mixing timeframes/strategies.",
                        }

        # If no proposal or hold, skip
        if not proposal or proposal.get("action") == "hold":
            return {
                "decision": ExecutionDecision.REJECT.value,
                "reasoning": "No active signal to evaluate",
            }

        # Check 5: Signal Deduplication
        # Prevents the same signal from executing multiple times within cooldown window
        is_duplicate, duplicate_reason = self._is_duplicate_signal(proposal)
        if is_duplicate:
            logger.warning(f"PORTFOLIO MANAGER: {duplicate_reason}")
            return {
                "decision": ExecutionDecision.REJECT.value,
                "approved": False,
                "reasoning": duplicate_reason,
            }

        # Calculate dynamic confidence threshold
        confidence_threshold = self._calculate_dynamic_threshold(regime)

        # Check signal confidence against threshold
        signal_confidence = proposal.get("confidence", 0.5)

        # Evaluate portfolio impact
        impact = self._evaluate_portfolio_impact(proposal)

        # Get key levels from context for stop loss optimization
        key_levels = context.get("key_levels", {})

        # Also add nearest S/R to key_levels if available
        if context.get("nearest_support"):
            if "supports" not in key_levels:
                key_levels["supports"] = []
            key_levels["nearest_support"] = context.get("nearest_support")
        if context.get("nearest_resistance"):
            if "resistances" not in key_levels:
                key_levels["resistances"] = []
            key_levels["nearest_resistance"] = context.get("nearest_resistance")

        # Get ATR from multiple sources (for dynamic stop calculation)
        atr = None
        # 1. Try to get from proposal (strategies like Turtle include n_value/ATR)
        if proposal:
            atr = proposal.get("atr") or proposal.get("n_value")
        # 2. Try to get from market data indicators
        if not atr and market_data:
            indicators = market_data.get("indicators", {})
            atr = indicators.get("atr") or indicators.get("atr_14")

        # Check if this is an LLM override - skip confidence threshold check
        is_llm_override = proposal.get("llm_override", False)

        # Make final decision with dynamic stop loss
        plan = self._make_execution_decision(
            proposal=proposal,
            signal_confidence=signal_confidence,
            confidence_threshold=confidence_threshold,
            impact=impact,
            regime=regime,
            key_levels=key_levels,
            atr=atr,
            skip_confidence_check=is_llm_override,  # LLM overrides skip confidence check
        )

        # Log AI decision
        ai_logger = get_ai_logger()
        await ai_logger.log_decision(
            stage=STAGE_RISK_ASSESSMENT,
            model="portfolio_manager_v1",
            input_data={
                "signal": proposal,
                "signal_confidence": signal_confidence,
                "portfolio_exposure": self._get_current_exposure(),
                "regime": regime,
            },
            output_data={
                "decision": plan.decision.value,
                "confidence_threshold": confidence_threshold,
                "adjusted_size": plan.adjusted_size,
                "portfolio_impact": plan.portfolio_impact,
            },
            explanation=plan.reasoning,
        )

        # Record executed signal for deduplication if approved
        is_approved = plan.decision in [ExecutionDecision.EXECUTE, ExecutionDecision.REDUCE_SIZE]
        if is_approved:
            self._record_executed_signal(proposal)

        # Include LLM reasoning in the result if available
        llm_reasoning = proposal.get("llm_reasoning", "") if proposal else ""
        llm_context = proposal.get("llm_context", "") if proposal else ""
        llm_risk = proposal.get("llm_risk", "") if proposal else ""

        return {
            "decision": plan.decision.value,
            "approved": is_approved,
            "adjusted_size": plan.adjusted_size,
            "adjusted_stop_price": plan.adjusted_stop_price,
            "adjusted_target_price": plan.adjusted_target_price,
            "confidence_threshold": confidence_threshold,
            "signal_confidence": signal_confidence,
            "reasoning": plan.reasoning,
            "portfolio_impact": plan.portfolio_impact,
            # LLM validation context
            "llm_reasoning": llm_reasoning,
            "llm_context": llm_context,
            "llm_risk": llm_risk,
            "llm_override": is_llm_override,
        }

    def _calculate_dynamic_threshold(self, regime: Dict[str, Any]) -> float:
        """Calculate dynamic confidence threshold based on conditions.

        Threshold increases (more conservative) when:
        - High volatility
        - Near max drawdown
        - Low recent win rate
        - High portfolio exposure

        Threshold decreases (more aggressive) when:
        - Low volatility with clear trend
        - Strong recent performance
        - Low portfolio exposure

        Args:
            regime: Current market regime

        Returns:
            Adjusted confidence threshold (0.0 - 1.0)
        """
        threshold = self.base_confidence_threshold

        # Volatility adjustment
        volatility = regime.get("volatility", "medium")
        if volatility == "high":
            threshold += 0.1  # More conservative in high vol
        elif volatility == "low":
            threshold -= 0.05  # Slightly more aggressive in low vol

        # Drawdown adjustment
        if self.portfolio.max_drawdown_today > self.max_daily_drawdown * 0.5:
            threshold += 0.15  # Much more conservative near drawdown limit
        elif self.portfolio.max_drawdown_today > self.max_daily_drawdown * 0.3:
            threshold += 0.08

        # Win rate adjustment
        if self.portfolio.win_rate_recent < 0.4:
            threshold += 0.1  # More conservative on losing streak
        elif self.portfolio.win_rate_recent > 0.6:
            threshold -= 0.05  # Slightly more aggressive on winning streak

        # Exposure adjustment
        current_exposure = self._get_current_exposure()
        if current_exposure > self.max_portfolio_exposure * 0.7:
            threshold += 0.1  # More selective when already exposed

        # Clamp to valid range
        return max(0.3, min(0.9, threshold))

    def _evaluate_portfolio_impact(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how the proposed trade impacts portfolio.

        Args:
            proposal: Trade proposal

        Returns:
            Impact analysis
        """
        symbol = proposal.get("symbol", "BTCUSDT")
        action = proposal.get("action", "hold")
        size = proposal.get("size", 0)

        # Current exposure
        current_exposure = self._get_current_exposure()
        new_exposure = current_exposure + (size / (self.portfolio.total_equity or 1))

        # Check correlation with existing positions
        correlated_exposure = self._get_correlated_exposure(symbol)

        # Check if this is adding to or reducing a position
        existing_position = self._get_position(symbol)
        is_adding = existing_position and (
            (existing_position.side == "long" and action == "buy") or
            (existing_position.side == "short" and action == "sell")
        )

        # Check if this is a hedge (opposite direction to existing)
        is_hedge = existing_position and not is_adding

        return {
            "current_exposure": current_exposure,
            "new_exposure": new_exposure,
            "correlated_exposure": correlated_exposure,
            "is_adding_to_position": is_adding,
            "is_hedge": is_hedge,
            "exposure_increase": new_exposure - current_exposure,
            "exceeds_max_exposure": new_exposure > self.max_portfolio_exposure,
            "exceeds_correlation_limit": correlated_exposure > self.max_correlated_exposure,
        }

    def _make_execution_decision(
        self,
        proposal: Dict[str, Any],
        signal_confidence: float,
        confidence_threshold: float,
        impact: Dict[str, Any],
        regime: Dict[str, Any],
        key_levels: Optional[Dict[str, Any]] = None,
        atr: Optional[float] = None,
        skip_confidence_check: bool = False,
    ) -> ExecutionPlan:
        """Make final execution decision with dynamic stop loss.

        Args:
            proposal: Trade proposal
            signal_confidence: Signal's confidence
            confidence_threshold: Dynamic threshold
            impact: Portfolio impact analysis
            regime: Market regime
            key_levels: Support/resistance levels from context
            atr: Average True Range for volatility-based stops
            skip_confidence_check: If True, skip confidence threshold check (for LLM overrides)

        Returns:
            ExecutionPlan with final decision
        """
        reasoning_parts = []
        original_size = proposal.get("size", 0)
        adjusted_size = original_size

        # Get original TP/SL
        entry_price = proposal.get("price", 0)
        original_stop = proposal.get("stop_price")
        original_target = proposal.get("target_price")
        action = proposal.get("action", "buy").lower()
        direction = "long" if action == "buy" else "short"

        # Rule 1: Confidence check (skip if LLM override)
        if signal_confidence < confidence_threshold and not skip_confidence_check:
            reasoning_parts.append(
                f"Signal confidence ({signal_confidence:.2f}) below threshold ({confidence_threshold:.2f})"
            )
            return ExecutionPlan(
                decision=ExecutionDecision.REJECT,
                original_signal=proposal,
                adjusted_size=0,
                adjusted_stop_price=None,
                adjusted_target_price=None,
                confidence_threshold_used=confidence_threshold,
                signal_confidence=signal_confidence,
                reasoning=". ".join(reasoning_parts) + ". REJECTED: Insufficient confidence.",
                portfolio_impact=impact,
            )
        elif skip_confidence_check:
            reasoning_parts.append(
                f"LLM override: bypassing confidence check (confidence {signal_confidence:.2f}, threshold {confidence_threshold:.2f})"
            )

        # Rule 2: Max exposure check
        if impact["exceeds_max_exposure"]:
            # Can we reduce size to fit?
            max_allowed = (self.max_portfolio_exposure - impact["current_exposure"]) * (self.portfolio.total_equity or 1)
            if max_allowed > original_size * 0.3:  # At least 30% of original
                adjusted_size = max_allowed
                reasoning_parts.append(
                    f"Reduced size from {original_size:.2f} to {adjusted_size:.2f} to stay within exposure limit"
                )
            else:
                reasoning_parts.append(
                    f"Would exceed max portfolio exposure ({self.max_portfolio_exposure:.0%})"
                )
                return ExecutionPlan(
                    decision=ExecutionDecision.REJECT,
                    original_signal=proposal,
                    adjusted_size=0,
                    adjusted_stop_price=None,
                    adjusted_target_price=None,
                    confidence_threshold_used=confidence_threshold,
                    signal_confidence=signal_confidence,
                    reasoning=". ".join(reasoning_parts) + ". REJECTED: Exposure limit.",
                    portfolio_impact=impact,
                )

        # Rule 3: Correlation check
        if impact["exceeds_correlation_limit"] and not impact["is_hedge"]:
            # Reduce size for correlated positions
            adjusted_size = adjusted_size * 0.5
            reasoning_parts.append(
                f"Reduced size by 50% due to correlation with existing positions"
            )

        # Rule 4: Drawdown protection
        if self.portfolio.max_drawdown_today > self.max_daily_drawdown:
            reasoning_parts.append(
                f"Daily drawdown limit ({self.max_daily_drawdown:.1%}) reached"
            )
            return ExecutionPlan(
                decision=ExecutionDecision.REJECT,
                original_signal=proposal,
                adjusted_size=0,
                adjusted_stop_price=None,
                adjusted_target_price=None,
                confidence_threshold_used=confidence_threshold,
                signal_confidence=signal_confidence,
                reasoning=". ".join(reasoning_parts) + ". REJECTED: Drawdown limit hit.",
                portfolio_impact=impact,
            )

        # Rule 5: Regime-specific size adjustments
        if regime.get("volatility") == "high":
            adjusted_size = adjusted_size * 0.7
            reasoning_parts.append("Reduced size by 30% due to high volatility")

        # =====================================================================
        # DYNAMIC STOP LOSS CALCULATION
        # This replaces the old fixed stop loss logic with confidence-scaled,
        # volatility-adaptive, key-level-aware stop loss calculation
        # =====================================================================
        if entry_price > 0:
            # Calculate dynamic stop loss and target
            sl_result = self.stop_loss_calculator.calculate(
                entry_price=entry_price,
                direction=direction,
                confidence=signal_confidence,
                atr=atr,
                regime_volatility=regime.get("volatility", "medium"),
                key_levels=key_levels,
                original_stop=original_stop,
                original_target=original_target,
            )

            adjusted_stop = sl_result.stop_price
            adjusted_target = sl_result.target_price

            # Log the dynamic stop loss details
            reasoning_parts.append(
                f"Dynamic SL: {sl_result.reasoning}"
            )
            reasoning_parts.append(
                f"R:R={sl_result.risk_reward_ratio:.2f}:1, "
                f"Stop={sl_result.stop_distance_pct:.2%}, "
                f"Target={sl_result.target_distance_pct:.2%}"
            )

            # Validate R:R is acceptable
            if sl_result.risk_reward_ratio < self.stop_loss_calculator.min_rr_ratio:
                reasoning_parts.append(
                    f"R:R ratio ({sl_result.risk_reward_ratio:.2f}) below minimum ({self.stop_loss_calculator.min_rr_ratio})"
                )
                # Still proceed but note the concern
        else:
            # No entry price - use original values
            adjusted_stop = original_stop
            adjusted_target = original_target
            reasoning_parts.append("No entry price - using original SL/TP")

        # All checks passed
        reasoning_parts.append(
            f"Confidence {signal_confidence:.2f} > threshold {confidence_threshold:.2f}"
        )

        # Determine final decision
        if adjusted_size < original_size * 0.5:
            decision = ExecutionDecision.REDUCE_SIZE
            reasoning_parts.append(f"Executing with reduced size: {adjusted_size:.2f}")
        else:
            decision = ExecutionDecision.EXECUTE
            reasoning_parts.append(f"Executing with size: {adjusted_size:.2f}")

        return ExecutionPlan(
            decision=decision,
            original_signal=proposal,
            adjusted_size=adjusted_size,
            adjusted_stop_price=adjusted_stop,
            adjusted_target_price=adjusted_target,
            confidence_threshold_used=confidence_threshold,
            signal_confidence=signal_confidence,
            reasoning=". ".join(reasoning_parts),
            portfolio_impact=impact,
        )

    def _get_current_exposure(self) -> float:
        """Get current portfolio exposure as fraction of equity."""
        if not self.portfolio.total_equity:
            return 0.0
        return self.portfolio.used_margin / self.portfolio.total_equity

    def _get_correlated_exposure(self, symbol: str) -> float:
        """Get exposure to assets correlated with given symbol."""
        # Find which correlation group this symbol belongs to
        symbol_group = None
        for group_name, symbols in self.correlation_groups.items():
            if symbol in symbols:
                symbol_group = group_name
                break

        if not symbol_group:
            return 0.0

        # Sum exposure to all symbols in the same group
        correlated_exposure = 0.0
        for position in self.portfolio.positions:
            if position.symbol in self.correlation_groups.get(symbol_group, []):
                correlated_exposure += position.margin_used

        if self.portfolio.total_equity:
            return correlated_exposure / self.portfolio.total_equity
        return 0.0

    def _get_position(self, symbol: str) -> Optional[Position]:
        """Get existing position for symbol."""
        for position in self.portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    def _update_portfolio_state(self, account_info: Dict[str, Any]):
        """Update portfolio state from account info.

        Args:
            account_info: Account information from exchange
        """
        self.portfolio.total_equity = float(account_info.get("equity", 0))
        self.portfolio.used_margin = float(account_info.get("usedMargin", 0))
        self.portfolio.available_margin = float(account_info.get("availableMargin", 0))
        self.portfolio.unrealized_pnl = float(account_info.get("unrealizedPnl", 0))

        # Update positions
        positions_data = account_info.get("positions", [])
        self.portfolio.positions = [
            Position(
                symbol=p.get("symbol", ""),
                side=p.get("side", "").lower(),
                size=float(p.get("size", 0)),
                entry_price=float(p.get("entryPrice", 0)),
                unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                margin_used=float(p.get("margin", 0)),
            )
            for p in positions_data
            if float(p.get("size", 0)) > 0
        ]

    def record_trade_result(self, trade: Dict[str, Any]):
        """Record a completed trade for win rate calculation.

        Args:
            trade: Completed trade with pnl
        """
        self.trade_history.append(trade)

        # Keep only recent trades
        if len(self.trade_history) > self.win_rate_lookback:
            self.trade_history = self.trade_history[-self.win_rate_lookback:]

        # Update win rate
        if self.trade_history:
            wins = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
            self.portfolio.win_rate_recent = wins / len(self.trade_history)

    def update_daily_drawdown(self, current_equity: float, starting_equity: float):
        """Update daily drawdown calculation.

        Args:
            current_equity: Current account equity
            starting_equity: Equity at start of day
        """
        if starting_equity > 0:
            drawdown = (starting_equity - current_equity) / starting_equity
            self.portfolio.max_drawdown_today = max(
                self.portfolio.max_drawdown_today,
                drawdown
            )

    # =========================================================================
    # SIGNAL DEDUPLICATION METHODS
    # =========================================================================

    def _generate_signal_signature(self, proposal: Dict[str, Any]) -> str:
        """Generate a unique signature hash for a trading signal.

        The signature is based on:
        - Symbol
        - Strategy
        - Direction (buy/sell)
        - Timeframe
        - Key reasoning elements (normalized)

        Args:
            proposal: Trade proposal

        Returns:
            SHA256 hash of the signal signature
        """
        symbol = proposal.get("symbol", "UNKNOWN")
        strategy = proposal.get("strategy", "unknown")
        action = proposal.get("action", "hold").lower()
        timeframe = proposal.get("timeframe", "4h")
        reasoning = proposal.get("reason", "") or proposal.get("explanation", "")

        # Normalize reasoning - extract key indicator values
        # This prevents minor price changes from creating new signatures
        reasoning_normalized = self._normalize_reasoning(reasoning)

        # Create signature string
        signature_parts = [
            symbol.upper(),
            strategy.lower(),
            action,
            timeframe.lower(),
            reasoning_normalized,
        ]
        signature_str = "|".join(signature_parts)

        # Generate hash
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]

    def _normalize_reasoning(self, reasoning: str) -> str:
        """Normalize reasoning to extract key signal components.

        Removes specific price values but keeps indicator states.

        Args:
            reasoning: Raw reasoning string

        Returns:
            Normalized reasoning string
        """
        if not reasoning:
            return "none"

        # Convert to lowercase
        text = reasoning.lower()

        # Extract key signal components (without specific values)
        components = []

        # RSI states
        if "rsi" in text:
            if "oversold" in text:
                components.append("rsi_oversold")
            elif "overbought" in text:
                components.append("rsi_overbought")
            else:
                components.append("rsi_signal")

        # Bollinger Band states
        if "bollinger" in text or "bb" in text:
            if "below" in text or "lower" in text:
                components.append("bb_lower")
            elif "above" in text or "upper" in text:
                components.append("bb_upper")

        # MACD states
        if "macd" in text:
            if "bullish" in text or "cross" in text:
                components.append("macd_bullish")
            elif "bearish" in text:
                components.append("macd_bearish")

        # Trend states
        if "trend" in text:
            if "uptrend" in text or "bullish" in text:
                components.append("trend_up")
            elif "downtrend" in text or "bearish" in text:
                components.append("trend_down")

        # Breakout signals
        if "breakout" in text:
            if "bullish" in text or "upside" in text:
                components.append("breakout_up")
            elif "bearish" in text or "downside" in text:
                components.append("breakout_down")
            else:
                components.append("breakout")

        # Pattern signals
        patterns = ["vcp", "head", "shoulder", "double", "flag", "wedge", "triangle"]
        for pattern in patterns:
            if pattern in text:
                components.append(f"pattern_{pattern}")

        # Support/Resistance
        if "support" in text:
            components.append("support")
        if "resistance" in text:
            components.append("resistance")

        # Pivot signals
        if "pivot" in text:
            components.append("pivot")

        # Momentum signals
        if "momentum" in text:
            components.append("momentum")

        # Volume signals
        if "volume" in text:
            if "surge" in text or "spike" in text:
                components.append("volume_surge")
            else:
                components.append("volume")

        # If no components found, use a hash of first 50 chars
        if not components:
            return hashlib.md5(text[:50].encode()).hexdigest()[:8]

        return "_".join(sorted(set(components)))

    def _is_duplicate_signal(self, proposal: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Check if a signal is a duplicate of a recently executed signal.

        Args:
            proposal: Trade proposal

        Returns:
            Tuple of (is_duplicate, reason_string)
        """
        signature_hash = self._generate_signal_signature(proposal)
        timeframe = proposal.get("timeframe", "4h").lower()
        cooldown = self._signal_cooldowns.get(timeframe, self._signal_cooldowns["default"])

        # Clean up old signatures first
        self._cleanup_old_signatures()

        # Check if this signature exists and is within cooldown
        if signature_hash in self._executed_signals:
            sig = self._executed_signals[signature_hash]
            time_since = datetime.now() - sig.executed_at
            remaining = cooldown - time_since

            if remaining.total_seconds() > 0:
                # Still in cooldown
                hours_remaining = remaining.total_seconds() / 3600
                return True, (
                    f"DUPLICATE SIGNAL BLOCKED: Same {sig.strategy} signal on {sig.symbol} "
                    f"({sig.direction}) executed {time_since.total_seconds()/60:.0f}m ago. "
                    f"Cooldown: {hours_remaining:.1f}h remaining. "
                    f"Reasoning: {sig.reasoning_summary[:50]}..."
                )

        return False, None

    def _record_executed_signal(self, proposal: Dict[str, Any]):
        """Record a signal as executed for deduplication tracking.

        Args:
            proposal: Trade proposal that was executed
        """
        signature_hash = self._generate_signal_signature(proposal)
        symbol = proposal.get("symbol", "UNKNOWN")
        strategy = proposal.get("strategy", "unknown")
        action = proposal.get("action", "hold")
        timeframe = proposal.get("timeframe", "4h")
        reasoning = proposal.get("reason", "") or proposal.get("explanation", "")

        self._executed_signals[signature_hash] = SignalSignature(
            signature_hash=signature_hash,
            symbol=symbol,
            strategy=strategy,
            direction=action,
            timeframe=timeframe,
            executed_at=datetime.now(),
            reasoning_summary=self._normalize_reasoning(reasoning),
        )

        logger.info(
            f"DEDUP: Recorded signal signature {signature_hash[:8]} | "
            f"{strategy} {symbol} {action} ({timeframe}) | "
            f"Cooldown: {self._signal_cooldowns.get(timeframe.lower(), self._signal_cooldowns['default'])}"
        )

    def _cleanup_old_signatures(self):
        """Remove signatures that are past their maximum cooldown period."""
        max_age = timedelta(hours=48)  # Keep signatures for max 48 hours
        now = datetime.now()

        expired = [
            sig_hash for sig_hash, sig in self._executed_signals.items()
            if (now - sig.executed_at) > max_age
        ]

        for sig_hash in expired:
            del self._executed_signals[sig_hash]

        if expired:
            logger.debug(f"DEDUP: Cleaned up {len(expired)} expired signal signatures")

    def get_active_signatures(self) -> List[Dict[str, Any]]:
        """Get list of currently active (non-expired) signal signatures.

        Returns:
            List of signature info dicts
        """
        self._cleanup_old_signatures()
        now = datetime.now()

        result = []
        for sig in self._executed_signals.values():
            timeframe = sig.timeframe.lower()
            cooldown = self._signal_cooldowns.get(timeframe, self._signal_cooldowns["default"])
            time_since = now - sig.executed_at
            remaining = cooldown - time_since

            if remaining.total_seconds() > 0:
                result.append({
                    "signature": sig.signature_hash[:8],
                    "symbol": sig.symbol,
                    "strategy": sig.strategy,
                    "direction": sig.direction,
                    "timeframe": sig.timeframe,
                    "executed_at": sig.executed_at.isoformat(),
                    "cooldown_remaining_hours": remaining.total_seconds() / 3600,
                    "reasoning": sig.reasoning_summary,
                })

        return result
