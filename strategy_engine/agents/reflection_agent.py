"""Reflection Agent for Position Review.

Reviews open positions and suggests actions based on:
1. Regime change since entry
2. Thesis validation
3. P&L vs targets
4. Time in trade
5. Portfolio exposure

Runs every hour (configurable) via PositionReviewLoop.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.memory import (
    TradeRecord,
    PositionAction,
    PositionReview,
)


class ReflectionAgent(BaseAgent):
    """Agent that reviews open positions and suggests actions.

    This agent is part of the self-evolving RL system, providing
    continuous position evaluation based on changing market conditions.
    """

    name = "reflection_agent"
    stage_name = "Position Reflection"
    model_name = "reflection-v1"

    # Default thresholds
    DEFAULT_CONFIG = {
        "regime_change_sensitivity": 0.7,  # How sensitive to regime changes
        "min_profit_to_add": 0.01,  # 1% profit before pyramiding
        "max_loss_before_review": -0.02,  # -2% triggers detailed review
        "max_hold_time_hours": 168,  # 7 days max hold
        "trailing_stop_activation": 0.03,  # 3% profit activates trailing
        "trailing_stop_distance": 0.015,  # 1.5% trailing distance
    }

    def __init__(self, config: Dict[str, Any], llm_analyzer=None):
        """Initialize Reflection Agent.

        Args:
            config: Agent configuration
            llm_analyzer: LLM analyzer for complex evaluations
        """
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        self.llm = llm_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process all open positions and return review results.

        Args:
            context: Must contain:
                - positions: List of open positions from exchange
                - trade_records: Dict mapping trade_id to TradeRecord
                - current_regime: Current market regime
                - market_data: Current market data

        Returns:
            Dictionary with:
                - reviews: List of PositionReview objects
                - actions_needed: List of positions needing action
                - explanation: Summary of reviews
        """
        positions = context.get("positions", [])
        trade_records = context.get("trade_records", {})
        current_regime = context.get("current_regime", {})
        market_data = context.get("market_data", {})

        reviews: List[PositionReview] = []
        actions_needed: List[PositionReview] = []

        for position in positions:
            trade_id = position.get("trade_id") or position.get("orderId")
            trade_record = trade_records.get(str(trade_id))

            review = await self._review_position(
                position=position,
                trade_record=trade_record,
                current_regime=current_regime,
                market_data=market_data,
            )

            reviews.append(review)

            if review.action != PositionAction.HOLD:
                actions_needed.append(review)
                logger.info(
                    f"Reflection: {review.symbol} -> {review.action.value} "
                    f"(confidence: {review.confidence:.0%}) - {review.reason}"
                )

        return {
            "reviews": reviews,
            "actions_needed": actions_needed,
            "explanation": self._generate_summary(reviews),
            "confidence": 1.0 if reviews else 0.0,
        }

    async def _review_position(
        self,
        position: Dict[str, Any],
        trade_record: Optional[TradeRecord],
        current_regime: Dict[str, str],
        market_data: Dict[str, Any],
    ) -> PositionReview:
        """Review a single position and suggest action.

        Args:
            position: Position data from exchange
            trade_record: Original trade record (if available)
            current_regime: Current market regime
            market_data: Current market data

        Returns:
            PositionReview with suggested action
        """
        symbol = position.get("symbol", "UNKNOWN")
        current_price = float(market_data.get("price", position.get("markPrice", 0)))

        # Extract position details
        entry_price = float(position.get("entryPrice", position.get("avgCost", 0)))
        position_size = float(position.get("total", position.get("size", 0)))
        side = position.get("holdSide", position.get("side", "long"))

        # Calculate P&L
        if side == "long":
            unrealized_pnl = (current_price - entry_price) * position_size
            pnl_pct = ((current_price - entry_price) / entry_price) if entry_price else 0
        else:
            unrealized_pnl = (entry_price - current_price) * position_size
            pnl_pct = ((entry_price - current_price) / entry_price) if entry_price else 0

        # Get entry regime and time
        entry_regime = {}
        entry_time = datetime.now()
        trade_id = str(position.get("trade_id", position.get("orderId", "")))

        if trade_record:
            entry_regime = trade_record.entry_regime
            entry_time = trade_record.entry_timestamp
            trade_id = trade_record.trade_id

        time_in_position = int((datetime.now() - entry_time).total_seconds())

        # Check for regime change
        regime_changed = self._check_regime_change(entry_regime, current_regime)

        # Base review object
        review = PositionReview(
            trade_id=trade_id,
            symbol=symbol,
            current_pnl=unrealized_pnl,
            current_pnl_pct=pnl_pct * 100,
            time_in_position=time_in_position,
            entry_regime=entry_regime,
            current_regime=current_regime,
            regime_changed=regime_changed,
            action=PositionAction.HOLD,
            confidence=0.5,
            reason="No action needed",
        )

        # Run checks in priority order
        action, confidence, reason, details = await self._evaluate_position(
            position=position,
            trade_record=trade_record,
            current_regime=current_regime,
            market_data=market_data,
            pnl_pct=pnl_pct,
            time_in_position=time_in_position,
            regime_changed=regime_changed,
        )

        review.action = action
        review.confidence = confidence
        review.reason = reason

        if details:
            review.suggested_exit_price = details.get("exit_price")
            review.suggested_stop = details.get("stop")
            review.suggested_size_change = details.get("size_change")

        return review

    async def _evaluate_position(
        self,
        position: Dict[str, Any],
        trade_record: Optional[TradeRecord],
        current_regime: Dict[str, str],
        market_data: Dict[str, Any],
        pnl_pct: float,
        time_in_position: int,
        regime_changed: bool,
    ) -> tuple[PositionAction, float, str, Optional[Dict]]:
        """Evaluate position and determine action.

        Returns:
            Tuple of (action, confidence, reason, details)
        """
        details = {}

        # 1. Check max hold time
        max_hold_seconds = self.config["max_hold_time_hours"] * 3600
        if time_in_position > max_hold_seconds:
            return (
                PositionAction.CLOSE,
                0.85,
                f"Max hold time exceeded ({time_in_position // 3600}h)",
                {"exit_price": market_data.get("price")},
            )

        # 2. Check severe loss
        if pnl_pct < self.config["max_loss_before_review"]:
            # Use LLM for detailed analysis
            if self.llm and trade_record:
                llm_assessment = await self._get_llm_assessment(
                    position, trade_record, current_regime, market_data, pnl_pct
                )
                if "close" in llm_assessment.lower():
                    return (
                        PositionAction.CLOSE,
                        0.8,
                        f"LLM recommends close: {llm_assessment[:100]}",
                        {"exit_price": market_data.get("price")},
                    )
                elif "reduce" in llm_assessment.lower():
                    return (
                        PositionAction.REDUCE,
                        0.7,
                        f"LLM recommends reduce: {llm_assessment[:100]}",
                        {"size_change": -0.5},  # Reduce by 50%
                    )

        # 3. Check regime change impact
        if regime_changed:
            # Regime change is concerning - evaluate thesis
            thesis_valid = self._check_thesis_validity(
                trade_record, current_regime, pnl_pct
            )

            if not thesis_valid:
                return (
                    PositionAction.CLOSE,
                    0.75,
                    "Regime changed and thesis no longer valid",
                    {"exit_price": market_data.get("price")},
                )
            else:
                # Regime changed but thesis still valid - just note it
                return (
                    PositionAction.HOLD,
                    0.6,
                    "Regime changed but thesis holds - monitoring",
                    None,
                )

        # 4. Check trailing stop opportunity
        if pnl_pct >= self.config["trailing_stop_activation"]:
            current_price = market_data.get("price", 0)
            trailing_distance = self.config["trailing_stop_distance"]

            side = position.get("holdSide", "long")
            if side == "long":
                new_stop = current_price * (1 - trailing_distance)
            else:
                new_stop = current_price * (1 + trailing_distance)

            # Check if we should update stop
            current_stop = position.get("stopLoss", 0)
            if side == "long" and (not current_stop or new_stop > current_stop):
                return (
                    PositionAction.ADJUST_STOP,
                    0.7,
                    f"Trailing stop: move to {new_stop:.2f}",
                    {"stop": new_stop},
                )
            elif side == "short" and (not current_stop or new_stop < current_stop):
                return (
                    PositionAction.ADJUST_STOP,
                    0.7,
                    f"Trailing stop: move to {new_stop:.2f}",
                    {"stop": new_stop},
                )

        # 5. Check pyramid opportunity
        if (
            pnl_pct >= self.config["min_profit_to_add"]
            and trade_record
            and self._should_pyramid(trade_record, current_regime, pnl_pct)
        ):
            return (
                PositionAction.ADD,
                0.65,
                "Position profitable, trend aligned - pyramid opportunity",
                {"size_change": 0.5},  # Add 50% to position
            )

        # Default: hold
        return (
            PositionAction.HOLD,
            0.6,
            "Position within parameters - holding",
            None,
        )

    def _check_regime_change(
        self,
        entry_regime: Dict[str, str],
        current_regime: Dict[str, str],
    ) -> bool:
        """Check if market regime has changed significantly.

        Args:
            entry_regime: Regime at trade entry
            current_regime: Current regime

        Returns:
            True if regime has changed significantly
        """
        if not entry_regime or not current_regime:
            return False

        changes = 0
        total = 0

        for key in ["volatility", "trend", "volume"]:
            if key in entry_regime and key in current_regime:
                total += 1
                if entry_regime[key] != current_regime[key]:
                    changes += 1

        if total == 0:
            return False

        change_ratio = changes / total
        return change_ratio >= self.config["regime_change_sensitivity"]

    def _check_thesis_validity(
        self,
        trade_record: Optional[TradeRecord],
        current_regime: Dict[str, str],
        pnl_pct: float,
    ) -> bool:
        """Check if original trade thesis is still valid.

        Args:
            trade_record: Original trade record
            current_regime: Current regime

        Returns:
            True if thesis is still valid
        """
        if not trade_record:
            return True  # Can't invalidate without history

        strategy = trade_record.entry_strategy

        # Strategy-specific thesis checks
        if strategy == "trend_following":
            # Trend following requires trend to persist
            if current_regime.get("trend") == "ranging":
                return False
            # If we're long and trend is now bearish, thesis broken
            if trade_record.entry_side == "long" and current_regime.get("trend") == "bearish":
                return False
            if trade_record.entry_side == "short" and current_regime.get("trend") == "bullish":
                return False

        elif strategy == "mean_reversion":
            # Mean reversion expects mean to exist
            if current_regime.get("volatility") == "extreme":
                return False  # Extreme volatility breaks mean reversion
            # If already reverted significantly, thesis complete
            if pnl_pct > 0.02:  # >2% profit
                return True  # Thesis played out

        elif strategy == "turtle_trading":
            # Turtle requires trending market
            if current_regime.get("trend") == "ranging":
                return False

        return True

    def _should_pyramid(
        self,
        trade_record: TradeRecord,
        current_regime: Dict[str, str],
        pnl_pct: float,
    ) -> bool:
        """Determine if we should pyramid (add to winning position).

        Args:
            trade_record: Original trade record
            current_regime: Current regime
            pnl_pct: Current P&L percentage

        Returns:
            True if pyramiding is recommended
        """
        strategy = trade_record.entry_strategy

        # Only pyramid trend-following strategies
        if strategy not in ["trend_following", "turtle_trading"]:
            return False

        # Require significant profit
        if pnl_pct < 0.02:  # 2% minimum
            return False

        # Require trend alignment
        side = trade_record.entry_side
        trend = current_regime.get("trend", "unknown")

        if side == "long" and trend != "bullish":
            return False
        if side == "short" and trend != "bearish":
            return False

        # Require reasonable volatility
        if current_regime.get("volatility") == "extreme":
            return False

        return True

    async def _get_llm_assessment(
        self,
        position: Dict[str, Any],
        trade_record: TradeRecord,
        current_regime: Dict[str, str],
        market_data: Dict[str, Any],
        pnl_pct: float,
    ) -> str:
        """Get LLM assessment for complex position evaluation.

        Args:
            position: Position data
            trade_record: Trade record
            current_regime: Current regime
            market_data: Current market data
            pnl_pct: Current P&L percentage

        Returns:
            LLM assessment text
        """
        if not self.llm:
            return "hold"  # Default without LLM

        prompt = f"""
Evaluate this open trading position:

**Position:**
- Symbol: {trade_record.symbol}
- Side: {trade_record.entry_side.upper()}
- Entry: ${trade_record.entry_price:.2f}
- Current: ${market_data.get('price', 0):.2f}
- P&L: {pnl_pct*100:.2f}%
- Time held: {(datetime.now() - trade_record.entry_timestamp).total_seconds() / 3600:.1f} hours

**Entry Reasoning:**
{trade_record.entry_reasoning}

**Entry Regime:**
- Volatility: {trade_record.entry_regime.get('volatility', 'unknown')}
- Trend: {trade_record.entry_regime.get('trend', 'unknown')}

**Current Regime:**
- Volatility: {current_regime.get('volatility', 'unknown')}
- Trend: {current_regime.get('trend', 'unknown')}

Should we:
1. HOLD - keep position as is
2. CLOSE - exit the position
3. REDUCE - reduce position size by 50%

Respond with one word (HOLD, CLOSE, or REDUCE) followed by a brief reason (under 50 words).
"""

        try:
            return await self.llm.analyze_market(
                symbol=trade_record.symbol,
                price=market_data.get("price", 0),
                regime=current_regime,
                additional_context=prompt,
            )
        except Exception as e:
            logger.error(f"LLM assessment failed: {e}")
            return "hold"

    def _generate_summary(self, reviews: List[PositionReview]) -> str:
        """Generate summary of all position reviews.

        Args:
            reviews: List of position reviews

        Returns:
            Summary string
        """
        if not reviews:
            return "No open positions to review"

        holds = [r for r in reviews if r.action == PositionAction.HOLD]
        closes = [r for r in reviews if r.action == PositionAction.CLOSE]
        reduces = [r for r in reviews if r.action == PositionAction.REDUCE]
        adds = [r for r in reviews if r.action == PositionAction.ADD]
        adjusts = [r for r in reviews if r.action == PositionAction.ADJUST_STOP]

        parts = [f"Reviewed {len(reviews)} position(s):"]

        if holds:
            parts.append(f"- HOLD: {len(holds)}")
        if closes:
            parts.append(f"- CLOSE: {len(closes)} ({', '.join(r.symbol for r in closes)})")
        if reduces:
            parts.append(f"- REDUCE: {len(reduces)} ({', '.join(r.symbol for r in reduces)})")
        if adds:
            parts.append(f"- ADD: {len(adds)} ({', '.join(r.symbol for r in adds)})")
        if adjusts:
            parts.append(f"- ADJUST STOP: {len(adjusts)}")

        return "\n".join(parts)

    async def review_single_position(
        self,
        position: Dict[str, Any],
        trade_record: Optional[TradeRecord],
        current_regime: Dict[str, str],
        market_data: Dict[str, Any],
    ) -> PositionReview:
        """Public method to review a single position.

        Args:
            position: Position data from exchange
            trade_record: Original trade record
            current_regime: Current market regime
            market_data: Current market data

        Returns:
            PositionReview with suggested action
        """
        return await self._review_position(
            position=position,
            trade_record=trade_record,
            current_regime=current_regime,
            market_data=market_data,
        )
