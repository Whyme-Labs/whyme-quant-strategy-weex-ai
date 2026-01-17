"""Discord webhook integration for trade notifications."""

import httpx
from datetime import datetime
from typing import Any, Dict, Optional
from loguru import logger


class DiscordNotifier:
    """Send formatted trade signals to Discord."""

    def __init__(
        self,
        webhook_url: str,
        bot_name: str = "WhyMe Quant Bot",
        bot_avatar: Optional[str] = None,
    ):
        """Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            bot_name: Name to display for the bot
            bot_avatar: Optional URL to bot avatar image
        """
        self.webhook_url = webhook_url
        self.bot_name = bot_name
        self.bot_avatar = bot_avatar
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def send_signal(
        self,
        signal_type: str,
        symbol: str,
        direction: str,
        price: float,
        size: float,
        confidence: float,
        strategy: str,
        reasoning: str,
        regime: Optional[Dict[str, str]] = None,
        llm_analysis: Optional[str] = None,
    ):
        """Send a trade signal notification to Discord.

        Args:
            signal_type: Type of signal (ENTRY, EXIT, ALERT)
            symbol: Trading pair
            direction: LONG or SHORT
            price: Entry/exit price
            size: Position size
            confidence: Signal confidence (0-1)
            strategy: Strategy name
            reasoning: Reasoning for the signal
            regime: Market regime info
            llm_analysis: LLM analysis text
        """
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return

        # Determine color based on signal
        if direction.upper() == "LONG":
            color = 0x00FF00  # Green
            emoji = "🟢"
        elif direction.upper() == "SHORT":
            color = 0xFF0000  # Red
            emoji = "🔴"
        else:
            color = 0xFFFF00  # Yellow
            emoji = "🟡"

        # Confidence bar
        confidence_pct = int(confidence * 100)
        filled = int(confidence * 10)
        conf_bar = "█" * filled + "░" * (10 - filled)

        # Build regime string
        regime_str = ""
        if regime:
            regime_str = f"**Volatility:** {regime.get('volatility', 'N/A')} | **Trend:** {regime.get('trend', 'N/A')} | **Volume:** {regime.get('volume', 'N/A')}"

        # Build embed
        embed = {
            "title": f"{emoji} {signal_type}: {direction.upper()} {symbol}",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "💰 Price",
                    "value": f"`${price:,.2f}`",
                    "inline": True,
                },
                {
                    "name": "📊 Size",
                    "value": f"`{size:.4f}`",
                    "inline": True,
                },
                {
                    "name": "🎯 Confidence",
                    "value": f"`{conf_bar}` {confidence_pct}%",
                    "inline": True,
                },
                {
                    "name": "📈 Strategy",
                    "value": f"`{strategy}`",
                    "inline": True,
                },
            ],
            "footer": {
                "text": "WhyMe Labs AI Strategy Engine | WEEX AI Wars",
            },
        }

        # Add regime if available
        if regime_str:
            embed["fields"].append({
                "name": "🌡️ Market Regime",
                "value": regime_str,
                "inline": False,
            })

        # Add reasoning
        embed["fields"].append({
            "name": "🧠 Reasoning",
            "value": reasoning[:500] if len(reasoning) > 500 else reasoning,
            "inline": False,
        })

        # Add LLM analysis if available
        if llm_analysis:
            embed["fields"].append({
                "name": "🤖 AI Analysis (DeepSeek)",
                "value": llm_analysis[:800] if len(llm_analysis) > 800 else llm_analysis,
                "inline": False,
            })

        payload = {
            "username": self.bot_name,
            "embeds": [embed],
        }
        if self.bot_avatar:
            payload["avatar_url"] = self.bot_avatar

        try:
            client = await self._get_client()
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()
            logger.debug(f"Discord notification sent: {signal_type} {direction} {symbol}")
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

    async def send_status(self, title: str, message: str, color: int = 0x3498DB):
        """Send a status update to Discord.

        Args:
            title: Status title
            message: Status message
            color: Embed color (default blue)
        """
        if not self.webhook_url:
            return

        embed = {
            "title": f"📢 {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "WhyMe Labs AI Strategy Engine",
            },
        }

        payload = {
            "username": self.bot_name,
            "embeds": [embed],
        }
        if self.bot_avatar:
            payload["avatar_url"] = self.bot_avatar

        try:
            client = await self._get_client()
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord status: {e}")

    async def send_error(self, error: str, context: Optional[str] = None):
        """Send an error notification to Discord.

        Args:
            error: Error message
            context: Additional context
        """
        if not self.webhook_url:
            return

        description = f"```\n{error}\n```"
        if context:
            description += f"\n**Context:** {context}"

        embed = {
            "title": "⚠️ Error Alert",
            "description": description,
            "color": 0xFF6B6B,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "WhyMe Labs AI Strategy Engine",
            },
        }

        payload = {
            "username": self.bot_name,
            "embeds": [embed],
        }
        if self.bot_avatar:
            payload["avatar_url"] = self.bot_avatar

        try:
            client = await self._get_client()
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord error: {e}")

    async def send_trace(self, stage: str, details: str, data: Optional[Dict[str, Any]] = None):
        """Send a debug trace message to Discord.

        Args:
            stage: Pipeline stage (e.g., "Regime Detection", "Signal Generated")
            details: Brief description
            data: Optional data to display
        """
        if not self.webhook_url:
            return

        # Stage-specific colors and emojis
        stage_config = {
            "regime": ("🌡️", 0x9B59B6),  # Purple
            "signal": ("📊", 0x3498DB),  # Blue
            "portfolio": ("💼", 0xF39C12),  # Orange
            "risk": ("🛡️", 0x1ABC9C),  # Teal
            "rejected": ("❌", 0x95A5A6),  # Gray
        }

        emoji, color = stage_config.get(stage.lower(), ("📋", 0x7F8C8D))

        description = details
        if data:
            data_str = "\n".join(f"• **{k}:** {v}" for k, v in data.items())
            description += f"\n\n{data_str}"

        embed = {
            "title": f"{emoji} {stage}",
            "description": description[:2000],
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Decision Trace | WhyMe Labs",
            },
        }

        payload = {
            "username": self.bot_name,
            "embeds": [embed],
        }
        if self.bot_avatar:
            payload["avatar_url"] = self.bot_avatar

        try:
            client = await self._get_client()
            response = await client.post(self.webhook_url, json=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord trace: {e}")


# Singleton instance
_notifier: Optional[DiscordNotifier] = None


def get_discord_notifier(webhook_url: Optional[str] = None) -> DiscordNotifier:
    """Get or create Discord notifier singleton.

    Args:
        webhook_url: Discord webhook URL (only needed on first call)

    Returns:
        DiscordNotifier instance
    """
    global _notifier
    if _notifier is None:
        from shared.config import get_settings
        settings = get_settings()
        url = webhook_url or settings.discord_webhook_url
        _notifier = DiscordNotifier(
            webhook_url=url,
            bot_name=settings.discord_bot_name,
            bot_avatar=settings.discord_bot_avatar or None,
        )
    return _notifier
