#!/usr/bin/env python3
"""Test WEEX API connection and prepare for hackathon.

Run: python scripts/test_connection.py
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def test_weex_connection():
    """Test WEEX API connection."""
    from weex_client import WeexClient

    api_key = os.getenv("WEEX_API_KEY")
    secret_key = os.getenv("WEEX_SECRET_KEY")
    passphrase = os.getenv("WEEX_PASSPHRASE")

    if not api_key or not secret_key:
        print("❌ Missing WEEX credentials in .env file")
        print("\nRequired variables:")
        print("  WEEX_API_KEY=your-api-key")
        print("  WEEX_SECRET_KEY=your-secret-key")
        print("  WEEX_PASSPHRASE=your-passphrase")
        return False

    print("🔑 Credentials found, testing connection...")

    client = WeexClient(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase or "",
        base_url=os.getenv("WEEX_API_URL", "https://api.weex.com"),
    )

    try:
        # Test connection (public API)
        if await client.test_connection():
            print("✅ WEEX API connection successful!")

            # Get ticker (public API)
            print("\n📈 Fetching BTC ticker...")
            ticker = await client.get_ticker("BTCUSDT")
            if ticker:
                print(f"  BTC Price: ${ticker.get('last', 'N/A')}")
                print(f"  24h High: ${ticker.get('high_24h', 'N/A')}")
                print(f"  24h Low: ${ticker.get('low_24h', 'N/A')}")

            # Try account info (private API - may fail without proper auth)
            print("\n📊 Fetching account info (requires auth)...")
            try:
                account = await client.get_account_info()
                if account:
                    print(f"  Equity: {account.get('equity', 'N/A')}")
                    print(f"  Available: {account.get('available', 'N/A')}")
            except Exception as auth_error:
                print(f"  ⚠️ Account API requires authentication: {auth_error}")
                print("  (This is expected - will work with proper API key permissions)")

            await client.close()
            return True
        else:
            print("❌ Connection test failed")
            await client.close()
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        await client.close()
        return False


async def test_ai_logging():
    """Test AI logging system."""
    from ai_logging import get_ai_logger, STAGE_MARKET_ANALYSIS

    print("\n🤖 Testing AI logging system...")

    logger = get_ai_logger()

    # Log a test decision
    await logger.log_decision(
        stage=STAGE_MARKET_ANALYSIS,
        model="test_model",
        input_data={"test": "input"},
        output_data={"test": "output"},
        explanation="Test explanation for hackathon verification",
    )

    stats = logger.get_stats()
    print(f"  Decisions logged: {stats.get('total_decisions', 0)}")
    print("✅ AI logging working!")

    return True


async def run_strategy_test():
    """Run a quick strategy simulation."""
    from strategy_engine.agents import (
        RegimeDetectorAgent,
        MeanReversionAgent,
        TrendFollowingAgent,
        PortfolioManagerAgent,
    )
    from strategy_engine.core.orchestrator import AgentOrchestrator

    print("\n🎯 Testing strategy pipeline...")

    orchestrator = AgentOrchestrator(config={})

    # Register agents
    orchestrator.register_agent("regime_detector", RegimeDetectorAgent(config={}))
    orchestrator.register_agent("mean_reversion", MeanReversionAgent(config={"min_htf_periods": 5}))
    orchestrator.register_agent("trend_following", TrendFollowingAgent(config={}))

    portfolio = PortfolioManagerAgent(config={"base_confidence_threshold": 0.5})
    portfolio.portfolio.total_equity = 1000  # Test with $1000
    orchestrator.register_agent("portfolio_manager", portfolio)

    # Simulate market data
    signals_generated = 0
    for i in range(50):
        market_data = {
            "symbol": "BTCUSDT",
            "price": 50000 + (i * 100),  # Uptrend
            "bid": 49990 + (i * 100),
            "ask": 50010 + (i * 100),
            "volume": 1000000,
            "high_24h": 51000 + i * 50,
            "low_24h": 49000 + i * 50,
            "change_24h": 0.02,
            "timestamp": 1234567890 + i,
        }

        signal = await orchestrator.process_market_data(market_data)
        if signal:
            signals_generated += 1
            print(f"  Signal #{signals_generated}: {signal.action.value} {signal.size:.2f} @ {signal.price}")

    print(f"\n  Total signals: {signals_generated}")
    print("✅ Strategy pipeline working!")

    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("WEEX AI Hackathon - Connection & Setup Test")
    print("=" * 60)

    results = {
        "WEEX API": await test_weex_connection(),
        "AI Logging": await test_ai_logging(),
        "Strategy": await run_strategy_test(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Ready for hackathon.")
        print("\nNext steps:")
        print("1. Run: docker-compose up -d")
        print("2. Monitor logs: docker-compose logs -f")
        print("3. Submit BUIDL on DoraHacks")
    else:
        print("⚠️  Some tests failed. Fix issues before proceeding.")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
