"""LLM integration via OpenRouter for AI-powered analysis."""

import httpx
import json
from typing import Any, Dict, List, Optional
from loguru import logger


class LLMAnalyzer:
    """AI-powered market analysis using LLM via OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek/deepseek-chat",
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        http_referer: str = "https://whymelabs.com",
        app_name: str = "WhyMe Quant Strategy Engine",
    ):
        """Initialize LLM analyzer.

        Args:
            api_key: OpenRouter API key
            model: Model to use
            api_url: OpenRouter API URL
            http_referer: HTTP referer for requests
            app_name: Application name for requests
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.http_referer = http_referer
        self.app_name = app_name
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def analyze_market(
        self,
        symbol: str,
        price: float,
        regime: Dict[str, str],
        signal: Optional[Dict[str, Any]] = None,
        recent_prices: Optional[List[float]] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """Analyze market conditions and provide trading insights.

        Args:
            symbol: Trading pair
            price: Current price
            regime: Market regime (volatility, trend, volume)
            signal: Signal from technical analysis
            recent_prices: Recent price history
            additional_context: Additional context for analysis

        Returns:
            AI analysis text
        """
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
            return "LLM analysis unavailable - API key not configured"

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(
            symbol, price, regime, signal, recent_prices, additional_context
        )

        try:
            client = await self._get_client()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_name,
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 500,
                "temperature": 0.3,  # Lower for more consistent analysis
            }

            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            logger.debug(f"LLM analysis completed for {symbol}")
            return content

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    async def analyze_signal(
        self,
        symbol: str,
        price: float,
        direction: str,
        strategy: str,
        confidence: float,
        reasoning: str,
        regime: Dict[str, str],
    ) -> str:
        """Analyze a trading signal and provide AI opinion.

        Args:
            symbol: Trading pair
            price: Current price
            direction: Signal direction (long/short)
            strategy: Strategy that generated the signal
            confidence: Signal confidence
            reasoning: Technical reasoning
            regime: Market regime

        Returns:
            AI analysis of the signal
        """
        if not self.api_key:
            return "LLM unavailable"

        prompt = f"""
Analyze this trading signal:

**Symbol:** {symbol}
**Price:** ${price:,.2f}
**Direction:** {direction.upper()}
**Strategy:** {strategy}
**Confidence:** {confidence*100:.0f}%

**Technical Reasoning:**
{reasoning}

**Market Regime:**
- Volatility: {regime.get('volatility', 'unknown')}
- Trend: {regime.get('trend', 'unknown')}
- Volume: {regime.get('volume', 'unknown')}

Provide:
1. Quick assessment (agree/disagree with signal)
2. Key risk factor to watch
3. Suggested position sizing adjustment (if any)
Keep response under 150 words.
"""

        try:
            client = await self._get_client()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_name,
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior crypto quantitative trader. Provide concise, actionable analysis. Be direct and specific.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 300,
                "temperature": 0.2,
            }

            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content

        except Exception as e:
            logger.error(f"Signal analysis failed: {e}")
            return f"Analysis unavailable: {str(e)}"

    def _get_system_prompt(self) -> str:
        """Get system prompt for market analysis."""
        return """You are an expert crypto quantitative trading analyst at WhyMe Labs.
Your role is to provide concise, actionable market analysis for BTC/USDT perpetual futures.

Guidelines:
- Be direct and specific
- Focus on actionable insights
- Consider multiple timeframes
- Acknowledge uncertainty when present
- Keep responses concise (under 200 words)
- Use bullet points for clarity

You are analyzing for the WEEX AI Trading Hackathon where the goal is to maximize risk-adjusted returns."""

    async def analyze_trade_outcome(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        side: str,
        pnl_pct: float,
        duration_hours: float,
        strategy: str,
        entry_regime: Dict[str, str],
        exit_regime: Dict[str, str],
        entry_reasoning: str,
        exit_reason: str,
    ) -> str:
        """Analyze a completed trade and generate reflection.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            side: Trade side (long/short)
            pnl_pct: P&L percentage
            duration_hours: Trade duration in hours
            strategy: Strategy that generated the trade
            entry_regime: Market regime at entry
            exit_regime: Market regime at exit
            entry_reasoning: Original reasoning for entry
            exit_reason: Reason for exit

        Returns:
            LLM reflection on the trade
        """
        if not self.api_key:
            return "LLM unavailable"

        prompt = f"""
Analyze this completed trade:

**Trade Summary:**
- Symbol: {symbol}
- Side: {side.upper()}
- Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f}
- P&L: {pnl_pct:+.2f}%
- Duration: {duration_hours:.1f} hours
- Exit Reason: {exit_reason}

**Strategy:** {strategy}
**Entry Reasoning:** {entry_reasoning}

**Entry Regime:**
- Volatility: {entry_regime.get('volatility', 'unknown')}
- Trend: {entry_regime.get('trend', 'unknown')}

**Exit Regime:**
- Volatility: {exit_regime.get('volatility', 'unknown')}
- Trend: {exit_regime.get('trend', 'unknown')}

Provide a brief reflection (2-3 sentences):
1. What went well or poorly?
2. One specific lesson for next time
"""

        try:
            client = await self._get_client()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_name,
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a trading coach reviewing completed trades. Be concise, specific, and actionable.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 250,
                "temperature": 0.3,
            }

            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Trade outcome analysis failed: {e}")
            return f"Analysis unavailable: {str(e)}"

    async def extract_trading_patterns(
        self,
        trades_summary: str,
        strategy_breakdown: str,
        regime_breakdown: str,
        worst_trades: str,
        best_trades: str,
    ) -> str:
        """Extract patterns from trading history for learning.

        Args:
            trades_summary: Summary of total trades, P&L, win rate
            strategy_breakdown: Performance by strategy
            regime_breakdown: Performance by regime
            worst_trades: Details of worst trades
            best_trades: Details of best trades

        Returns:
            LLM analysis of patterns
        """
        if not self.api_key:
            return "LLM unavailable"

        prompt = f"""
Analyze trading patterns from recent performance:

**PERFORMANCE SUMMARY:**
{trades_summary}

**BY STRATEGY:**
{strategy_breakdown}

**BY REGIME:**
{regime_breakdown}

**WORST TRADES:**
{worst_trades}

**BEST TRADES:**
{best_trades}

Identify 2-3 specific, actionable patterns:
1. What market conditions lead to losses?
2. What conditions lead to wins?
3. Which strategy/regime combinations should be avoided or prioritized?

For each pattern:
PATTERN: [name]
DESCRIPTION: [what was observed]
RECOMMENDATION: [what to do differently]
"""

        try:
            client = await self._get_client()

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.http_referer,
                "X-Title": self.app_name,
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a quantitative trading analyst identifying patterns in trade performance. Focus on actionable insights.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 500,
                "temperature": 0.3,
            }

            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return f"Analysis unavailable: {str(e)}"

    def _build_analysis_prompt(
        self,
        symbol: str,
        price: float,
        regime: Dict[str, str],
        signal: Optional[Dict[str, Any]],
        recent_prices: Optional[List[float]],
        additional_context: Optional[str],
    ) -> str:
        """Build the analysis prompt.

        Args:
            symbol: Trading pair
            price: Current price
            regime: Market regime
            signal: Technical signal
            recent_prices: Price history
            additional_context: Extra context

        Returns:
            Formatted prompt string
        """
        parts = [f"**Market Analysis Request for {symbol}**\n"]
        parts.append(f"**Current Price:** ${price:,.2f}")

        # Regime info
        parts.append("\n**Market Regime:**")
        parts.append(f"- Volatility: {regime.get('volatility', 'unknown')}")
        parts.append(f"- Trend: {regime.get('trend', 'unknown')}")
        parts.append(f"- Volume: {regime.get('volume', 'unknown')}")

        # Signal info
        if signal:
            parts.append("\n**Technical Signal:**")
            parts.append(f"- Direction: {signal.get('direction', 'N/A')}")
            parts.append(f"- Strategy: {signal.get('strategy', 'N/A')}")
            parts.append(f"- Confidence: {signal.get('confidence', 0)*100:.0f}%")

        # Price context
        if recent_prices and len(recent_prices) >= 5:
            price_change = (price - recent_prices[0]) / recent_prices[0] * 100
            high = max(recent_prices)
            low = min(recent_prices)
            parts.append(f"\n**Recent Price Action:**")
            parts.append(f"- Change: {price_change:+.2f}%")
            parts.append(f"- Range: ${low:,.2f} - ${high:,.2f}")

        if additional_context:
            parts.append(f"\n**Additional Context:**\n{additional_context}")

        parts.append("\n\nProvide a brief analysis with:")
        parts.append("1. Market sentiment assessment")
        parts.append("2. Key levels to watch")
        parts.append("3. Recommended bias (long/short/neutral)")

        return "\n".join(parts)


# Singleton instance
_analyzer: Optional[LLMAnalyzer] = None


def get_llm_analyzer(api_key: Optional[str] = None, model: Optional[str] = None) -> LLMAnalyzer:
    """Get or create LLM analyzer singleton.

    Args:
        api_key: OpenRouter API key (only needed on first call)
        model: Model to use (only needed on first call)

    Returns:
        LLMAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        from shared.config import get_settings
        settings = get_settings()
        _analyzer = LLMAnalyzer(
            api_key=api_key or settings.openrouter_api_key,
            model=model or settings.llm_model,
            api_url=settings.openrouter_api_url,
            http_referer=settings.llm_http_referer,
            app_name=settings.llm_app_name,
        )
    return _analyzer
