#!/usr/bin/env python3
"""Test Discord webhook and LLM integration.

Run: python scripts/test_discord_llm.py
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def test_discord():
    """Test Discord webhook integration."""
    from shared.discord import get_discord_notifier

    print("\n" + "=" * 60)
    print("Testing Discord Webhook")
    print("=" * 60)

    discord = get_discord_notifier()

    if not discord.webhook_url:
        print("Discord webhook URL not configured")
        return False

    print("Sending test status message...")
    await discord.send_status(
        "Test Alert",
        "This is a test message from WhyMe Quant AI Strategy Engine.\n"
        "**Status:** All systems operational\n"
        "**Mode:** Test mode",
        color=0x3498DB,  # Blue
    )
    print("Status message sent!")

    print("\nSending test signal...")
    await discord.send_signal(
        signal_type="TEST SIGNAL",
        symbol="BTCUSDT",
        direction="long",
        price=104500.00,
        size=0.001,
        confidence=0.75,
        strategy="mean_reversion",
        reasoning="RSI oversold at 28.5, price below lower Bollinger Band. Testing Discord integration.",
        regime={
            "volatility": "medium",
            "trend": "sideways",
            "volume": "normal",
        },
        llm_analysis="This is a test LLM analysis. In production, DeepSeek will provide real-time market insights and signal validation.",
    )
    print("Signal message sent!")

    await discord.close()
    print("\nDiscord test completed successfully!")
    return True


async def test_llm():
    """Test LLM integration via OpenRouter."""
    from shared.llm import get_llm_analyzer

    print("\n" + "=" * 60)
    print("Testing LLM Integration (DeepSeek via OpenRouter)")
    print("=" * 60)

    llm = get_llm_analyzer()

    if not llm.api_key:
        print("OpenRouter API key not configured")
        return False

    print(f"Model: {llm.model}")
    print("\nSending test analysis request...")

    analysis = await llm.analyze_signal(
        symbol="BTCUSDT",
        price=104500.00,
        direction="long",
        strategy="mean_reversion",
        confidence=0.75,
        reasoning="RSI oversold at 28.5, price below lower Bollinger Band. High volume spike detected.",
        regime={
            "volatility": "medium",
            "trend": "sideways",
            "volume": "high",
        },
    )

    print("\n" + "-" * 40)
    print("LLM Analysis Response:")
    print("-" * 40)
    print(analysis)
    print("-" * 40)

    await llm.close()
    print("\nLLM test completed successfully!")
    return True


async def test_combined():
    """Test combined Discord + LLM flow."""
    from shared.discord import get_discord_notifier
    from shared.llm import get_llm_analyzer

    print("\n" + "=" * 60)
    print("Testing Combined Discord + LLM Flow")
    print("=" * 60)

    discord = get_discord_notifier()
    llm = get_llm_analyzer()

    # Get real analysis from LLM
    print("Getting LLM analysis...")
    analysis = await llm.analyze_signal(
        symbol="BTCUSDT",
        price=104500.00,
        direction="long",
        strategy="trend_following",
        confidence=0.82,
        reasoning="Strong breakout above 20-day high with volume confirmation. EMA alignment bullish.",
        regime={
            "volatility": "high",
            "trend": "strong_up",
            "volume": "high",
        },
    )

    # Send to Discord with real LLM analysis
    print("Sending to Discord with LLM analysis...")
    await discord.send_signal(
        signal_type="LIVE SIGNAL",
        symbol="BTCUSDT",
        direction="long",
        price=104500.00,
        size=0.002,
        confidence=0.82,
        strategy="trend_following",
        reasoning="Strong breakout above 20-day high with volume confirmation. EMA alignment bullish.",
        regime={
            "volatility": "high",
            "trend": "strong_up",
            "volume": "high",
        },
        llm_analysis=analysis,
    )

    await discord.close()
    await llm.close()

    print("\nCombined test completed successfully!")
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("WhyMe Quant - Discord & LLM Integration Test")
    print("=" * 60)

    results = {
        "Discord Webhook": await test_discord(),
        "LLM (DeepSeek)": await test_llm(),
        "Combined Flow": await test_combined(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Discord and LLM integration working.")
        print("\nCheck your Discord channel for the test messages!")
    else:
        print("Some tests failed. Check configuration.")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
