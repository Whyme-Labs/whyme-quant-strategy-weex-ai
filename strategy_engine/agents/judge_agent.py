"""Judge Agent for Trade Scoring.

Scores completed trades using multi-objective reward function:
1. Return Score: Risk-adjusted return (Sharpe-like)
2. Execution Score: Slippage and fill quality
3. Timing Score: Entry/exit timing quality
4. Risk Management Score: Stop adherence, position sizing
5. Discipline Score: Following the plan

Runs on every trade close via TradeOutcomeLoop.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

from .base_agent import BaseAgent
from ..models.memory import TradeRecord, TradeScore


class JudgeAgent(BaseAgent):
    """Agent that scores completed trades on multiple objectives.

    Provides quantitative evaluation of trade quality to feed
    into the learning system.
    """

    name = "judge_agent"
    stage_name = "Trade Evaluation"
    model_name = "judge-v1"

    # Multi-objective scoring weights
    DEFAULT_WEIGHTS = {
        "return": 0.35,       # Risk-adjusted return
        "execution": 0.15,    # Slippage and fill quality
        "timing": 0.15,       # Entry/exit timing
        "risk_management": 0.25,  # Stop adherence, sizing
        "discipline": 0.10,   # Following the plan
    }

    # Scoring parameters
    DEFAULT_CONFIG = {
        # Return scoring
        "target_sharpe": 2.0,  # Target Sharpe ratio for 100 score
        "min_return_for_good": 0.01,  # 1% return is "good"
        "max_loss_penalty": -0.05,  # Losses beyond 5% heavily penalized

        # Execution scoring
        "acceptable_slippage": 0.001,  # 0.1% slippage is acceptable
        "bad_slippage": 0.005,  # 0.5% slippage is bad

        # Timing scoring
        "good_mfe_capture": 0.7,  # 70% of MFE captured is good
        "good_mae_control": 0.3,  # MAE < 30% of MFE is good

        # Risk management scoring
        "expected_risk_reward": 2.0,  # Expected R:R ratio
        "stop_adherence_bonus": 10,  # Bonus for following stops

        # Discipline scoring
        "confidence_threshold": 0.6,  # Trades below this penalized
    }

    def __init__(self, config: Dict[str, Any], llm_analyzer=None):
        """Initialize Judge Agent.

        Args:
            config: Agent configuration
            llm_analyzer: LLM for qualitative assessment
        """
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        self.weights = config.get("weights", self.DEFAULT_WEIGHTS)
        self.llm = llm_analyzer

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a completed trade and return score.

        Args:
            context: Must contain:
                - trade: TradeRecord of completed trade

        Returns:
            Dictionary with:
                - score: TradeScore object
                - explanation: Detailed explanation
        """
        trade = context.get("trade")
        if not trade:
            return {
                "score": None,
                "explanation": "No trade provided",
                "confidence": 0.0,
            }

        score = await self.score_trade(trade)

        return {
            "score": score,
            "explanation": score.explanation,
            "confidence": 1.0,
        }

    async def score_trade(self, trade: TradeRecord) -> TradeScore:
        """Score a completed trade on multiple objectives.

        Args:
            trade: Completed TradeRecord

        Returns:
            TradeScore with detailed breakdown
        """
        # Calculate component scores
        return_score = self._score_return(trade)
        execution_score = self._score_execution(trade)
        timing_score = self._score_timing(trade)
        risk_score = self._score_risk_management(trade)
        discipline_score = self._score_discipline(trade)

        # Calculate weighted total
        total_score = (
            return_score * self.weights["return"]
            + execution_score * self.weights["execution"]
            + timing_score * self.weights["timing"]
            + risk_score * self.weights["risk_management"]
            + discipline_score * self.weights["discipline"]
        )

        # Generate explanation
        explanation = await self._generate_explanation(
            trade=trade,
            scores={
                "return": return_score,
                "execution": execution_score,
                "timing": timing_score,
                "risk_management": risk_score,
                "discipline": discipline_score,
            },
            total=total_score,
        )

        return TradeScore(
            trade_id=trade.trade_id,
            total_score=total_score,
            return_score=return_score,
            execution_score=execution_score,
            timing_score=timing_score,
            risk_management_score=risk_score,
            discipline_score=discipline_score,
            weights=self.weights,
            explanation=explanation,
        )

    def _score_return(self, trade: TradeRecord) -> float:
        """Score the trade's return.

        Uses a Sharpe-like risk-adjusted measure.

        Args:
            trade: Completed trade

        Returns:
            Score 0-100
        """
        if trade.pnl_pct is None:
            return 50.0  # Neutral if no P&L

        pnl_pct = trade.pnl_pct / 100  # Convert to decimal

        # Winning trade scoring
        if pnl_pct > 0:
            # Scale: 0% = 50, target_min = 70, 2x target = 90, 3x+ = 100
            target = self.config["min_return_for_good"]
            if pnl_pct >= target * 3:
                return 100.0
            elif pnl_pct >= target * 2:
                return 90.0 + (pnl_pct - target * 2) / target * 10
            elif pnl_pct >= target:
                return 70.0 + (pnl_pct - target) / target * 20
            else:
                return 50.0 + pnl_pct / target * 20

        # Losing trade scoring
        else:
            max_penalty = self.config["max_loss_penalty"]
            if pnl_pct <= max_penalty:
                return 0.0  # Complete failure
            else:
                # Scale from 50 at 0% to 0 at max_penalty
                return max(0, 50.0 * (1 - abs(pnl_pct) / abs(max_penalty)))

    def _score_execution(self, trade: TradeRecord) -> float:
        """Score trade execution quality.

        Measures slippage and fill quality.

        Args:
            trade: Completed trade

        Returns:
            Score 0-100
        """
        # Without detailed execution data, estimate from outcomes
        # Future: integrate with actual fill data

        # If we have MFE/MAE, we can infer execution quality
        if trade.max_favorable_excursion is not None and trade.pnl is not None:
            # Good execution: actual P&L close to MFE (captured the move)
            if trade.max_favorable_excursion > 0:
                capture_ratio = trade.pnl / trade.max_favorable_excursion
                capture_ratio = max(0, min(1, capture_ratio))
                return 50.0 + capture_ratio * 50.0
            else:
                # Never went positive - execution didn't matter
                return 50.0

        # Default: neutral score without data
        return 60.0  # Assume decent execution

    def _score_timing(self, trade: TradeRecord) -> float:
        """Score entry/exit timing quality.

        Measures how well we captured available move.

        Args:
            trade: Completed trade

        Returns:
            Score 0-100
        """
        mfe = trade.max_favorable_excursion
        mae = trade.max_adverse_excursion
        pnl = trade.pnl

        if mfe is None or mae is None or pnl is None:
            return 50.0  # Neutral without data

        # Calculate MFE capture ratio
        if mfe > 0:
            mfe_capture = pnl / mfe
            mfe_capture = max(0, min(1, mfe_capture))
        else:
            mfe_capture = 0

        # Calculate MAE control (lower is better)
        if mfe > 0:
            mae_ratio = abs(mae) / mfe if mfe != 0 else 1
        else:
            mae_ratio = 1

        # Score MFE capture (0-60 points)
        good_capture = self.config["good_mfe_capture"]
        if mfe_capture >= good_capture:
            mfe_score = 50 + (mfe_capture - good_capture) / (1 - good_capture) * 10
        else:
            mfe_score = mfe_capture / good_capture * 50

        # Score MAE control (0-40 points)
        good_mae = self.config["good_mae_control"]
        if mae_ratio <= good_mae:
            mae_score = 40
        elif mae_ratio >= 1:
            mae_score = 0
        else:
            mae_score = 40 * (1 - (mae_ratio - good_mae) / (1 - good_mae))

        return min(100, mfe_score + mae_score)

    def _score_risk_management(self, trade: TradeRecord) -> float:
        """Score risk management quality.

        Measures stop adherence, position sizing, and risk/reward.

        Args:
            trade: Completed trade

        Returns:
            Score 0-100
        """
        score = 50.0  # Start neutral

        # Bonus for proper exit via stop loss
        if trade.exit_reason:
            from ..models.memory import ExitReason
            if trade.exit_reason == ExitReason.STOP_LOSS:
                # Hit stop loss is good risk management (controlled loss)
                if trade.pnl_pct and trade.pnl_pct > -3:  # Small controlled loss
                    score += 20
                else:
                    score += 10
            elif trade.exit_reason == ExitReason.TAKE_PROFIT:
                score += 25  # Hit target is great
            elif trade.exit_reason == ExitReason.TRAILING_STOP:
                score += 20  # Trailing stop is good

        # Penalize positions that went too negative
        if trade.max_adverse_excursion is not None:
            mae_pct = trade.max_adverse_excursion / trade.entry_price * 100 if trade.entry_price else 0
            if abs(mae_pct) > 5:  # >5% drawdown
                score -= 15
            elif abs(mae_pct) > 3:  # >3% drawdown
                score -= 5

        # Bonus for good risk/reward (if we can calculate it)
        if trade.pnl_pct and trade.max_adverse_excursion:
            actual_rr = trade.pnl_pct / abs(trade.max_adverse_excursion) if trade.max_adverse_excursion else 0
            expected_rr = self.config["expected_risk_reward"]
            if actual_rr >= expected_rr:
                score += 15
            elif actual_rr > 1:
                score += 10

        return max(0, min(100, score))

    def _score_discipline(self, trade: TradeRecord) -> float:
        """Score trading discipline.

        Measures if trader followed the plan.

        Args:
            trade: Completed trade

        Returns:
            Score 0-100
        """
        score = 50.0  # Start neutral

        # Higher confidence entries score better
        threshold = self.config["confidence_threshold"]
        if trade.entry_confidence >= threshold:
            # Scale from threshold (50) to 1.0 (80)
            confidence_score = 50 + (trade.entry_confidence - threshold) / (1 - threshold) * 30
            score = max(score, confidence_score)
        else:
            # Low confidence trade - penalize
            score -= 20

        # Bonus if strategy matches regime (if we can verify)
        if trade.entry_regime and trade.entry_strategy:
            strategy = trade.entry_strategy
            regime = trade.entry_regime

            # Check strategy-regime alignment
            if strategy == "trend_following" and regime.get("trend") in ["bullish", "bearish"]:
                score += 15  # Good: trend following in trending market
            elif strategy == "mean_reversion" and regime.get("trend") == "ranging":
                score += 15  # Good: mean reversion in ranging market
            elif strategy == "turtle_trading" and regime.get("trend") in ["bullish", "bearish"]:
                score += 15  # Good: turtle in trending market

            # Penalize misalignment
            if strategy == "trend_following" and regime.get("trend") == "ranging":
                score -= 10  # Bad: trend following in ranging market
            elif strategy == "mean_reversion" and regime.get("volatility") == "extreme":
                score -= 10  # Bad: mean reversion in extreme volatility

        return max(0, min(100, score))

    async def _generate_explanation(
        self,
        trade: TradeRecord,
        scores: Dict[str, float],
        total: float,
    ) -> str:
        """Generate detailed explanation of the score.

        Args:
            trade: The trade being scored
            scores: Component scores
            total: Total score

        Returns:
            Explanation string
        """
        parts = [
            f"Trade Score: {total:.1f}/100",
            "",
            "Component Scores:",
        ]

        for component, score in scores.items():
            weight = self.weights.get(component, 0)
            weighted = score * weight
            parts.append(f"  - {component.replace('_', ' ').title()}: {score:.1f} (weighted: {weighted:.1f})")

        parts.append("")

        # Add context
        if trade.pnl_pct is not None:
            result = "WIN" if trade.pnl_pct > 0 else "LOSS"
            parts.append(f"Result: {result} ({trade.pnl_pct:+.2f}%)")

        if trade.duration_seconds:
            hours = trade.duration_seconds / 3600
            parts.append(f"Duration: {hours:.1f} hours")

        if trade.exit_reason:
            parts.append(f"Exit: {trade.exit_reason.value}")

        # Add qualitative assessment if LLM available
        if self.llm:
            try:
                llm_assessment = await self._get_llm_qualitative(trade, scores, total)
                if llm_assessment:
                    parts.append("")
                    parts.append(f"Analysis: {llm_assessment}")
            except Exception as e:
                logger.debug(f"LLM assessment skipped: {e}")

        return "\n".join(parts)

    async def _get_llm_qualitative(
        self,
        trade: TradeRecord,
        scores: Dict[str, float],
        total: float,
    ) -> str:
        """Get LLM qualitative assessment of trade.

        Args:
            trade: Trade being evaluated
            scores: Component scores
            total: Total score

        Returns:
            LLM assessment
        """
        prompt = f"""
Provide a brief (2-3 sentence) qualitative assessment of this trade:

Trade: {trade.symbol} {trade.entry_side.upper()}
Entry: ${trade.entry_price:.2f} -> Exit: ${trade.exit_price:.2f}
P&L: {trade.pnl_pct:+.2f}%
Strategy: {trade.entry_strategy}
Score: {total:.1f}/100

What was done well? What could be improved?
"""

        try:
            return await self.llm.analyze_market(
                symbol=trade.symbol,
                price=trade.exit_price or trade.entry_price,
                regime=trade.exit_regime or trade.entry_regime,
                additional_context=prompt,
            )
        except Exception as e:
            logger.error(f"LLM qualitative assessment failed: {e}")
            return ""

    async def score_batch(self, trades: List[TradeRecord]) -> List[TradeScore]:
        """Score multiple trades.

        Args:
            trades: List of completed trades

        Returns:
            List of TradeScores
        """
        scores = []
        for trade in trades:
            score = await self.score_trade(trade)
            scores.append(score)
        return scores

    def calculate_aggregate_metrics(
        self,
        scores: List[TradeScore],
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics from multiple scores.

        Args:
            scores: List of trade scores

        Returns:
            Dictionary of aggregate metrics
        """
        if not scores:
            return {}

        total_scores = [s.total_score for s in scores]
        return_scores = [s.return_score for s in scores]
        timing_scores = [s.timing_score for s in scores]
        risk_scores = [s.risk_management_score for s in scores]

        return {
            "count": len(scores),
            "avg_total": sum(total_scores) / len(total_scores),
            "min_total": min(total_scores),
            "max_total": max(total_scores),
            "avg_return": sum(return_scores) / len(return_scores),
            "avg_timing": sum(timing_scores) / len(timing_scores),
            "avg_risk": sum(risk_scores) / len(risk_scores),
            "scores_above_70": len([s for s in total_scores if s >= 70]),
            "scores_below_40": len([s for s in total_scores if s < 40]),
        }
