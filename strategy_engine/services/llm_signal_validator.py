"""LLM Signal Validator Service.

Integrates LLM (DeepSeek/MiMo via OpenRouter) into the trading decision pipeline
with FULL override authority. The LLM serves 3 roles:

1. **Signal Validator** - Analyze strategy signals before Portfolio Manager
2. **Reasoning Enhancer** - Add market context and reasoning to decisions
3. **Full Decision Maker** - Can approve rejected signals OR reject approved signals

Philosophy:
- LLM has final say on whether a signal should trade
- Rule-based systems provide the candidates, LLM makes the decision
- LLM can see market context that rules cannot quantify
- LLM adds human-like reasoning to every trade decision
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from shared.llm import LLMAnalyzer


class LLMDecision(Enum):
    """LLM decision on a trading signal."""
    APPROVE = "approve"                # LLM approves the signal
    REJECT = "reject"                  # LLM rejects the signal
    OVERRIDE_APPROVE = "override_approve"  # LLM approves a previously rejected signal
    DEFER = "defer"                    # LLM defers to rule-based system


@dataclass
class LLMValidationResult:
    """Result of LLM signal validation."""
    decision: LLMDecision
    confidence_adjustment: float  # -0.3 to +0.3 adjustment
    reasoning: str
    market_context: str
    risk_assessment: str
    key_factors: List[str]
    override_reason: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class ValidationCacheEntry:
    """Cache entry for recent validations."""
    result: LLMValidationResult
    timestamp: datetime
    symbol: str


class LLMSignalValidator:
    """LLM-powered signal validator with full override authority.

    This validator sits between strategy signals and the Portfolio Manager,
    providing intelligent analysis and decision-making beyond rule-based logic.

    Capabilities:
    - Validate signals using comprehensive market context
    - Adjust confidence based on LLM assessment
    - Reject signals that rules would approve (if LLM sees risk)
    - Approve signals that rules rejected (if LLM sees opportunity)
    - Add rich reasoning to every decision for trade journal
    """

    # Rate limiting: minimum seconds between validations per symbol
    VALIDATION_COOLDOWN_SECONDS = 60

    # Cache TTL for validation results
    CACHE_TTL_SECONDS = 300  # 5 minutes

    # Default confidence adjustment bounds
    MIN_CONFIDENCE_ADJUSTMENT = -0.3
    MAX_CONFIDENCE_ADJUSTMENT = 0.3

    def __init__(
        self,
        llm_analyzer: "LLMAnalyzer",
        validation_cooldown: int = 60,
        enable_override: bool = True,
    ):
        """Initialize LLM Signal Validator.

        Args:
            llm_analyzer: LLMAnalyzer instance for making LLM calls
            validation_cooldown: Minimum seconds between validations per symbol
            enable_override: Whether to allow LLM to override rejected signals
        """
        self.llm = llm_analyzer
        self.validation_cooldown = validation_cooldown
        self.enable_override = enable_override

        # Validation cache: symbol -> ValidationCacheEntry
        self._validation_cache: Dict[str, ValidationCacheEntry] = {}

        # Last validation time per symbol for rate limiting
        self._last_validation: Dict[str, datetime] = {}

        # Validation history for performance tracking
        self._validation_history: List[Dict[str, Any]] = []

    async def validate_signal(
        self,
        signal: Dict[str, Any],
        market_context: Dict[str, Any],
        alpha_result: Optional[Dict[str, Any]] = None,
        edge_result: Optional[List[Any]] = None,
        key_levels: Optional[Dict[str, Any]] = None,
        smc_data: Optional[Dict[str, Any]] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMValidationResult:
        """Validate a trading signal using LLM analysis.

        This is the primary validation method called for every strategy signal.

        Args:
            signal: Trading signal from strategy (contains action, confidence, etc.)
            market_context: Current market data and regime info
            alpha_result: Optional AlphaGenerator confirmation result
            edge_result: Optional EdgeScanner signals
            key_levels: Optional support/resistance levels
            smc_data: Optional Smart Money Concepts data
            recent_trades: Optional list of recent trades for context

        Returns:
            LLMValidationResult with decision, confidence adjustment, and reasoning
        """
        symbol = signal.get("symbol", "BTCUSDT")

        # Check rate limiting
        if not self._should_validate(symbol):
            logger.debug(f"LLM validation skipped for {symbol}: within cooldown period")
            return LLMValidationResult(
                decision=LLMDecision.DEFER,
                confidence_adjustment=0.0,
                reasoning="Rate limited - using rule-based decision",
                market_context="",
                risk_assessment="",
                key_factors=[],
            )

        try:
            # Build comprehensive prompt
            prompt = self._build_validation_prompt(
                signal=signal,
                market_context=market_context,
                alpha_result=alpha_result,
                edge_result=edge_result,
                key_levels=key_levels,
                smc_data=smc_data,
                recent_trades=recent_trades,
            )

            # Call LLM
            response = await self._call_llm_validation(prompt)

            # Parse response
            result = self._parse_llm_response(response)

            # Update tracking
            self._last_validation[symbol] = datetime.now()
            self._cache_result(symbol, result)
            self._record_validation(signal, result)

            logger.info(
                f"LLM Validation [{symbol}]: {result.decision.value} | "
                f"confidence_adj={result.confidence_adjustment:+.2f} | "
                f"{result.reasoning[:100]}..."
            )

            return result

        except Exception as e:
            logger.warning(f"LLM validation failed for {symbol}: {e}")
            return LLMValidationResult(
                decision=LLMDecision.DEFER,
                confidence_adjustment=0.0,
                reasoning=f"LLM unavailable ({str(e)[:50]}), deferring to rules",
                market_context="",
                risk_assessment="",
                key_factors=[],
            )

    async def review_rejected_signal(
        self,
        signal: Dict[str, Any],
        rejection_reason: str,
        market_context: Dict[str, Any],
    ) -> LLMValidationResult:
        """Review a signal that was rejected by the rule-based system.

        LLM can override and approve if it sees merit that rules missed.

        Args:
            signal: The rejected trading signal
            rejection_reason: Why the rule-based system rejected it
            market_context: Current market data and regime info

        Returns:
            LLMValidationResult with override decision if applicable
        """
        if not self.enable_override:
            return LLMValidationResult(
                decision=LLMDecision.DEFER,
                confidence_adjustment=0.0,
                reasoning="Override disabled",
                market_context="",
                risk_assessment="",
                key_factors=[],
            )

        symbol = signal.get("symbol", "BTCUSDT")

        try:
            prompt = self._build_override_review_prompt(
                signal=signal,
                rejection_reason=rejection_reason,
                market_context=market_context,
            )

            response = await self._call_llm_validation(prompt)
            result = self._parse_override_response(response)

            if result.decision == LLMDecision.OVERRIDE_APPROVE:
                logger.info(
                    f"LLM OVERRIDE [{symbol}]: Approving rejected signal | "
                    f"Original rejection: {rejection_reason[:50]}... | "
                    f"Override reason: {result.override_reason}"
                )

            return result

        except Exception as e:
            logger.warning(f"LLM override review failed for {symbol}: {e}")
            return LLMValidationResult(
                decision=LLMDecision.DEFER,
                confidence_adjustment=0.0,
                reasoning=f"LLM override review failed: {str(e)[:50]}",
                market_context="",
                risk_assessment="",
                key_factors=[],
            )

    def _should_validate(self, symbol: str) -> bool:
        """Check if we should validate (rate limiting).

        Args:
            symbol: Trading symbol

        Returns:
            True if validation should proceed
        """
        if symbol not in self._last_validation:
            return True

        time_since = datetime.now() - self._last_validation[symbol]
        return time_since.total_seconds() >= self.validation_cooldown

    def _cache_result(self, symbol: str, result: LLMValidationResult):
        """Cache validation result.

        Args:
            symbol: Trading symbol
            result: Validation result to cache
        """
        self._validation_cache[symbol] = ValidationCacheEntry(
            result=result,
            timestamp=datetime.now(),
            symbol=symbol,
        )

        # Cleanup old cache entries
        self._cleanup_cache()

    def _cleanup_cache(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired = [
            symbol for symbol, entry in self._validation_cache.items()
            if (now - entry.timestamp).total_seconds() > self.CACHE_TTL_SECONDS
        ]
        for symbol in expired:
            del self._validation_cache[symbol]

    def _record_validation(self, signal: Dict[str, Any], result: LLMValidationResult):
        """Record validation for performance tracking.

        Args:
            signal: Original signal
            result: Validation result
        """
        self._validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": signal.get("symbol", "UNKNOWN"),
            "strategy": signal.get("strategy", "unknown"),
            "original_confidence": signal.get("confidence", 0),
            "decision": result.decision.value,
            "confidence_adjustment": result.confidence_adjustment,
            "key_factors": result.key_factors,
        })

        # Keep only last 100 validations
        if len(self._validation_history) > 100:
            self._validation_history = self._validation_history[-100:]

    async def _call_llm_validation(self, prompt: str) -> str:
        """Call LLM with validation prompt.

        Args:
            prompt: Full prompt for validation

        Returns:
            Raw LLM response text
        """
        if not self.llm.api_key:
            raise ValueError("LLM API key not configured")

        client = await self.llm._get_client()

        headers = {
            "Authorization": f"Bearer {self.llm.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.llm.http_referer,
            "X-Title": self.llm.app_name,
        }

        system_prompt = """You are an expert quantitative trading analyst at WhyMe Labs.
Your role is to review trading signals and decide whether to approve, reject, or adjust them.

You have FULL override authority - you can:
1. APPROVE signals that look good
2. REJECT signals that are too risky or poorly timed
3. ADJUST confidence up or down based on your analysis

Be direct and decisive. Provide specific reasoning.
Always respond in valid JSON format as specified in the prompt."""

        payload = {
            "model": self.llm.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.2,  # Low temperature for more consistent decisions
        }

        response = await client.post(
            self.llm.api_url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _build_validation_prompt(
        self,
        signal: Dict[str, Any],
        market_context: Dict[str, Any],
        alpha_result: Optional[Dict[str, Any]] = None,
        edge_result: Optional[List[Any]] = None,
        key_levels: Optional[Dict[str, Any]] = None,
        smc_data: Optional[Dict[str, Any]] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build comprehensive prompt for signal validation.

        Args:
            signal: Trading signal
            market_context: Market data and regime
            alpha_result: AlphaGenerator result
            edge_result: EdgeScanner signals
            key_levels: Support/resistance levels
            smc_data: Smart Money Concepts data
            recent_trades: Recent trade history

        Returns:
            Formatted prompt string
        """
        # Extract signal details
        symbol = signal.get("symbol", "BTCUSDT")
        action = signal.get("action", "hold").upper()
        strategy = signal.get("strategy", "unknown")
        confidence = signal.get("confidence", 0.5)
        entry_price = signal.get("price", 0)
        stop_loss = signal.get("stop_price", 0)
        take_profit = signal.get("target_price", 0)
        timeframe = signal.get("timeframe", "4h")
        reasoning = signal.get("reason", signal.get("explanation", "No reasoning provided"))

        # Extract market context
        market_data = market_context.get("market_data", {})
        current_price = market_data.get("price", entry_price)
        change_24h = market_data.get("change_24h", 0)
        volume = market_data.get("volume", 0)

        regime = market_context.get("regime", {})
        volatility_regime = regime.get("volatility", "unknown")
        trend_regime = regime.get("trend", "unknown")
        volume_regime = regime.get("volume", "unknown")

        # Build prompt
        parts = [
            "## SIGNAL TO VALIDATE",
            f"- **Symbol:** {symbol}",
            f"- **Action:** {action}",
            f"- **Strategy:** {strategy}",
            f"- **Confidence:** {confidence*100:.0f}%",
            f"- **Timeframe:** {timeframe}",
            f"- **Entry Price:** ${entry_price:,.2f}" if entry_price else "- **Entry Price:** Not set",
            f"- **Stop Loss:** ${stop_loss:,.2f}" if stop_loss else "- **Stop Loss:** Not set",
            f"- **Take Profit:** ${take_profit:,.2f}" if take_profit else "- **Take Profit:** Not set",
            "",
            "## STRATEGY REASONING",
            reasoning[:500],
            "",
            "## MARKET CONTEXT",
            f"- **Current Price:** ${current_price:,.2f}" if current_price else "- **Current Price:** Unknown",
            f"- **24h Change:** {change_24h:+.2f}%",
            f"- **Volume:** {volume:,.0f}",
            f"- **Volatility Regime:** {volatility_regime}",
            f"- **Trend Regime:** {trend_regime}",
            f"- **Volume Regime:** {volume_regime}",
        ]

        # Add Alpha Generator context
        if alpha_result:
            alpha_direction = getattr(alpha_result, 'direction', 'N/A') if hasattr(alpha_result, 'direction') else alpha_result.get('direction', 'N/A')
            alpha_score = getattr(alpha_result, 'alpha', 0) if hasattr(alpha_result, 'alpha') else alpha_result.get('alpha', 0)
            alpha_conf = getattr(alpha_result, 'confidence', 0) if hasattr(alpha_result, 'confidence') else alpha_result.get('confidence', 0)
            parts.extend([
                "",
                "## ALPHA GENERATOR",
                f"- Direction: {alpha_direction}",
                f"- Alpha Score: {alpha_score:+.2f}" if isinstance(alpha_score, (int, float)) else f"- Alpha Score: {alpha_score}",
                f"- Confidence: {alpha_conf:.0%}" if isinstance(alpha_conf, (int, float)) else f"- Confidence: {alpha_conf}",
            ])

        # Add Edge Scanner context
        if edge_result and len(edge_result) > 0:
            parts.extend(["", "## EDGE SIGNALS"])
            for i, edge in enumerate(edge_result[:3]):
                edge_id = getattr(edge, 'edge_id', 'unknown') if hasattr(edge, 'edge_id') else edge.get('edge_id', 'unknown')
                edge_side = getattr(edge, 'side', 'unknown') if hasattr(edge, 'side') else edge.get('side', 'unknown')
                parts.append(f"- Edge {i+1}: {edge_id} ({edge_side})")

        # Add Key Levels context
        if key_levels:
            supports = key_levels.get("supports", [])
            resistances = key_levels.get("resistances", [])
            nearest_support = key_levels.get("nearest_support")
            nearest_resistance = key_levels.get("nearest_resistance")

            parts.extend(["", "## KEY LEVELS"])
            if nearest_support:
                price = nearest_support.get("price", 0) if isinstance(nearest_support, dict) else nearest_support
                parts.append(f"- Nearest Support: ${price:,.2f}" if price else "- Nearest Support: None")
            if nearest_resistance:
                price = nearest_resistance.get("price", 0) if isinstance(nearest_resistance, dict) else nearest_resistance
                parts.append(f"- Nearest Resistance: ${price:,.2f}" if price else "- Nearest Resistance: None")
            parts.append(f"- Total Supports: {len(supports)}")
            parts.append(f"- Total Resistances: {len(resistances)}")

        # Add SMC context
        if smc_data:
            smc_summary = market_context.get("smc_summary", {})
            trend = smc_summary.get("trend", "unknown")
            ob_count = smc_summary.get("active_ob_count", 0)
            fvg_count = smc_summary.get("active_fvg_count", 0)
            prem_disc = smc_summary.get("premium_discount", {})
            zone = prem_disc.get("current_zone", "unknown") if prem_disc else "unknown"

            parts.extend([
                "",
                "## SMART MONEY CONCEPTS",
                f"- Market Structure: {trend}",
                f"- Active Order Blocks: {ob_count}",
                f"- Active FVGs: {fvg_count}",
                f"- Premium/Discount Zone: {zone}",
            ])

        # Add recent trades context
        if recent_trades:
            wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            total = len(recent_trades)
            win_rate = (wins / total * 100) if total > 0 else 0

            parts.extend([
                "",
                "## RECENT PERFORMANCE",
                f"- Last {total} trades for {symbol}",
                f"- Win rate: {win_rate:.0f}%",
            ])

        # Add task and response format
        parts.extend([
            "",
            "## YOUR TASK",
            "Analyze this signal and provide your decision.",
            "",
            "Consider:",
            "1. Does the strategy reasoning make sense given current market conditions?",
            "2. Is the risk-reward acceptable?",
            "3. Are there any red flags or concerns?",
            "4. Does the technical context support this trade?",
            "",
            "Respond in JSON format:",
            "```json",
            "{",
            '    "decision": "approve" | "reject" | "defer",',
            '    "confidence_adjustment": -0.3 to +0.3,',
            '    "reasoning": "Brief explanation (2-3 sentences)",',
            '    "market_context": "Key market observation",',
            '    "risk_assessment": "Risk analysis",',
            '    "key_factors": ["factor1", "factor2", "factor3"]',
            "}",
            "```",
        ])

        return "\n".join(parts)

    def _build_override_review_prompt(
        self,
        signal: Dict[str, Any],
        rejection_reason: str,
        market_context: Dict[str, Any],
    ) -> str:
        """Build prompt for reviewing rejected signals.

        Args:
            signal: Rejected signal
            rejection_reason: Why it was rejected
            market_context: Market data

        Returns:
            Formatted prompt string
        """
        symbol = signal.get("symbol", "BTCUSDT")
        action = signal.get("action", "hold").upper()
        strategy = signal.get("strategy", "unknown")
        confidence = signal.get("confidence", 0.5)

        market_data = market_context.get("market_data", {})
        current_price = market_data.get("price", 0)

        regime = market_context.get("regime", {})
        volatility = regime.get("volatility", "unknown")
        trend = regime.get("trend", "unknown")

        return f"""## REJECTED SIGNAL FOR REVIEW

**Signal Details:**
- Symbol: {symbol}
- Action: {action}
- Strategy: {strategy}
- Confidence: {confidence*100:.0f}%

**Rejection Reason:**
{rejection_reason}

**Current Market Context:**
- Price: ${current_price:,.2f}
- Volatility: {volatility}
- Trend: {trend}

## YOUR TASK

Determine if this rejection should be OVERRIDDEN.

Consider:
1. Is the rejection reason valid in current market conditions?
2. Are there compelling factors that outweigh the rejection reason?
3. Is there an edge the rule-based system cannot see?
4. Would you take this trade as a professional trader?

Respond in JSON:
```json
{{
    "decision": "override_approve" | "defer",
    "confidence_adjustment": 0.0 to +0.15,
    "reasoning": "Why override (or why not)",
    "market_context": "Key observation",
    "risk_assessment": "Risk if overriding",
    "key_factors": ["factor1", "factor2"],
    "override_reason": "Specific reason for override (if approving)"
}}
```
"""

    def _parse_llm_response(self, response: str) -> LLMValidationResult:
        """Parse LLM response into validation result.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed LLMValidationResult
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group())

            # Parse decision
            decision_str = data.get("decision", "defer").lower()
            decision_map = {
                "approve": LLMDecision.APPROVE,
                "reject": LLMDecision.REJECT,
                "override_approve": LLMDecision.OVERRIDE_APPROVE,
                "defer": LLMDecision.DEFER,
            }
            decision = decision_map.get(decision_str, LLMDecision.DEFER)

            # Parse confidence adjustment with bounds
            conf_adj = float(data.get("confidence_adjustment", 0))
            conf_adj = max(self.MIN_CONFIDENCE_ADJUSTMENT,
                          min(self.MAX_CONFIDENCE_ADJUSTMENT, conf_adj))

            return LLMValidationResult(
                decision=decision,
                confidence_adjustment=conf_adj,
                reasoning=data.get("reasoning", "No reasoning provided"),
                market_context=data.get("market_context", ""),
                risk_assessment=data.get("risk_assessment", ""),
                key_factors=data.get("key_factors", []),
                override_reason=data.get("override_reason"),
                raw_response=response,
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Return defer decision on parse failure
            return LLMValidationResult(
                decision=LLMDecision.DEFER,
                confidence_adjustment=0.0,
                reasoning=f"Failed to parse LLM response: {str(e)[:50]}",
                market_context="",
                risk_assessment="",
                key_factors=[],
                raw_response=response,
            )

    def _parse_override_response(self, response: str) -> LLMValidationResult:
        """Parse override review response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed LLMValidationResult
        """
        result = self._parse_llm_response(response)

        # For override, ensure minimum confidence if approving
        if result.decision == LLMDecision.OVERRIDE_APPROVE:
            # Minimum 0.55 confidence for overrides
            if result.confidence_adjustment < 0.05:
                result = LLMValidationResult(
                    decision=result.decision,
                    confidence_adjustment=0.05,  # Minimum boost for override
                    reasoning=result.reasoning,
                    market_context=result.market_context,
                    risk_assessment=result.risk_assessment,
                    key_factors=result.key_factors,
                    override_reason=result.override_reason,
                    raw_response=result.raw_response,
                )

        return result

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring.

        Returns:
            Dictionary of validation stats
        """
        if not self._validation_history:
            return {
                "total_validations": 0,
                "approve_rate": 0,
                "reject_rate": 0,
                "defer_rate": 0,
                "avg_confidence_adjustment": 0,
            }

        total = len(self._validation_history)
        approves = sum(1 for v in self._validation_history if v["decision"] == "approve")
        rejects = sum(1 for v in self._validation_history if v["decision"] == "reject")
        defers = sum(1 for v in self._validation_history if v["decision"] == "defer")
        overrides = sum(1 for v in self._validation_history if v["decision"] == "override_approve")

        avg_adj = sum(v["confidence_adjustment"] for v in self._validation_history) / total

        return {
            "total_validations": total,
            "approve_rate": approves / total,
            "reject_rate": rejects / total,
            "defer_rate": defers / total,
            "override_rate": overrides / total,
            "avg_confidence_adjustment": avg_adj,
        }

    def get_cached_result(self, symbol: str) -> Optional[LLMValidationResult]:
        """Get cached validation result if available and not expired.

        Args:
            symbol: Trading symbol

        Returns:
            Cached result or None
        """
        if symbol not in self._validation_cache:
            return None

        entry = self._validation_cache[symbol]
        age = (datetime.now() - entry.timestamp).total_seconds()

        if age > self.CACHE_TTL_SECONDS:
            del self._validation_cache[symbol]
            return None

        return entry.result
