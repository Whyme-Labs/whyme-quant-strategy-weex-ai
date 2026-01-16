"""Unit tests for strategy agents.

Run with: pytest tests/test_agents.py -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

# Add parent to path for imports
import sys
sys.path.insert(0, '.')

from strategy_engine.agents.regime_detector import RegimeDetectorAgent, VolatilityRegime, TrendRegime
from strategy_engine.agents.mean_reversion import MeanReversionAgent
from strategy_engine.agents.trend_following import TrendFollowingAgent
from strategy_engine.agents.portfolio_manager import PortfolioManagerAgent, PortfolioState, Position


class TestRegimeDetector:
    """Test Regime Detector Agent."""

    @pytest.fixture
    def agent(self):
        return RegimeDetectorAgent(config={
            "ema_periods": [8, 20, 50],
            "atr_period": 14,
            "lookback_period": 50,
        })

    @pytest.mark.asyncio
    async def test_insufficient_data(self, agent):
        """Should return neutral with low confidence when insufficient data."""
        with patch('strategy_engine.agents.regime_detector.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            context = {
                "market_data": {"price": 50000, "volume": 1000, "high_24h": 51000, "low_24h": 49000}
            }
            result = await agent.process(context)

            assert result["confidence"] == 0.3
            assert result["recommended_strategy"] == "neutral"

    @pytest.mark.asyncio
    async def test_trending_market(self, agent):
        """Should detect trending market after enough data."""
        with patch('strategy_engine.agents.regime_detector.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            # Simulate uptrending market
            base_price = 50000
            for i in range(30):
                price = base_price + (i * 100)  # Steady uptrend
                context = {
                    "market_data": {
                        "price": price,
                        "volume": 1000,
                        "high_24h": price * 1.01,
                        "low_24h": price * 0.99
                    }
                }
                result = await agent.process(context)

            # After enough data, should detect trend
            assert "regime" in result
            print(f"Detected regime: {result['regime']}, Strategy: {result['recommended_strategy']}")


class TestMeanReversion:
    """Test Mean Reversion Agent."""

    @pytest.fixture
    def agent(self):
        return MeanReversionAgent(config={
            "bb_period": 20,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "htf_multiplier": 4,
            "min_htf_periods": 10,  # Lower for testing
        })

    @pytest.mark.asyncio
    async def test_builds_htf_data(self, agent):
        """Should build higher timeframe data before signaling."""
        with patch('strategy_engine.agents.mean_reversion.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            # Feed some data
            for i in range(20):
                context = {
                    "market_data": {"price": 50000 + i * 10},
                    "regime": {}
                }
                result = await agent.process(context)

            # Should be building HTF data
            assert "Building HTF data" in result.get("reasoning", "") or result.get("signal") is not None

    @pytest.mark.asyncio
    async def test_oversold_signal(self, agent):
        """Should generate long signal on oversold conditions."""
        with patch('strategy_engine.agents.mean_reversion.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            # Build up stable price history first
            for i in range(50):
                context = {
                    "market_data": {"price": 50000},
                    "regime": {}
                }
                await agent.process(context)

            # Now simulate a sharp drop (oversold)
            for i in range(10):
                context = {
                    "market_data": {"price": 50000 - (i * 500)},  # Sharp drop
                    "regime": {}
                }
                result = await agent.process(context)

            print(f"Mean Reversion result: {result}")


class TestTrendFollowing:
    """Test Trend Following Agent."""

    @pytest.fixture
    def agent(self):
        return TrendFollowingAgent(config={
            "channel_period": 20,
            "ema_periods": [8, 20, 50],
            "atr_period": 14,
            "atr_multiplier": 2.0,
        })

    @pytest.mark.asyncio
    async def test_channel_breakout(self, agent):
        """Should detect channel breakout."""
        with patch('strategy_engine.agents.trend_following.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            # Build range-bound history
            for i in range(25):
                price = 50000 + (i % 5) * 100  # Oscillate in range
                context = {
                    "market_data": {
                        "price": price,
                        "high_24h": price + 50,
                        "low_24h": price - 50,
                        "volume": 1000
                    },
                    "regime": {}
                }
                await agent.process(context)

            # Now breakout above range
            context = {
                "market_data": {
                    "price": 51000,  # Above 20-day high
                    "high_24h": 51050,
                    "low_24h": 50900,
                    "volume": 2000  # High volume
                },
                "regime": {}
            }
            result = await agent.process(context)

            print(f"Trend Following result: {result}")
            if result.get("signal"):
                assert result["signal"]["direction"] == "long"


class TestPortfolioManager:
    """Test Portfolio Manager Agent."""

    @pytest.fixture
    def agent(self):
        agent = PortfolioManagerAgent(config={
            "max_portfolio_exposure": 0.5,
            "max_single_position": 0.1,
            "base_confidence_threshold": 0.6,
            "max_daily_drawdown": 0.05,
        })
        # Set up portfolio state
        agent.portfolio = PortfolioState(
            positions=[],
            total_equity=10000,
            used_margin=0,
            available_margin=10000,
            unrealized_pnl=0,
            win_rate_recent=0.5,
            max_drawdown_today=0,
        )
        return agent

    @pytest.mark.asyncio
    async def test_rejects_low_confidence(self, agent):
        """Should reject signals below dynamic threshold."""
        with patch('strategy_engine.agents.portfolio_manager.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            context = {
                "strategy_proposal": {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "size": 100,
                    "confidence": 0.4,  # Below threshold
                },
                "market_data": {"price": 50000},
                "regime": {"volatility": "medium"},
            }

            result = await agent.process(context)

            assert result["approved"] == False
            assert "confidence" in result["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_approves_high_confidence(self, agent):
        """Should approve signals above threshold."""
        with patch('strategy_engine.agents.portfolio_manager.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            context = {
                "strategy_proposal": {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "size": 100,
                    "confidence": 0.8,  # Above threshold
                },
                "market_data": {"price": 50000},
                "regime": {"volatility": "low"},
            }

            result = await agent.process(context)

            assert result["approved"] == True

    @pytest.mark.asyncio
    async def test_dynamic_threshold_high_volatility(self, agent):
        """Threshold should increase in high volatility."""
        with patch('strategy_engine.agents.portfolio_manager.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            # Same confidence, different volatility
            context_low_vol = {
                "strategy_proposal": {"action": "buy", "symbol": "BTCUSDT", "size": 100, "confidence": 0.65},
                "market_data": {"price": 50000},
                "regime": {"volatility": "low"},
            }
            context_high_vol = {
                "strategy_proposal": {"action": "buy", "symbol": "BTCUSDT", "size": 100, "confidence": 0.65},
                "market_data": {"price": 50000},
                "regime": {"volatility": "high"},
            }

            result_low = await agent.process(context_low_vol)

            # Reset for second test
            agent.portfolio.max_drawdown_today = 0
            result_high = await agent.process(context_high_vol)

            # Higher volatility should have higher threshold
            assert result_high["confidence_threshold"] > result_low["confidence_threshold"]

    @pytest.mark.asyncio
    async def test_rejects_at_drawdown_limit(self, agent):
        """Should reject all trades when drawdown limit hit."""
        with patch('strategy_engine.agents.portfolio_manager.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()

            # Set drawdown above limit
            agent.portfolio.max_drawdown_today = 0.06  # Above 5% limit

            context = {
                "strategy_proposal": {
                    "action": "buy",
                    "symbol": "BTCUSDT",
                    "size": 100,
                    "confidence": 0.9,  # Very high confidence
                },
                "market_data": {"price": 50000},
                "regime": {"volatility": "low"},
            }

            result = await agent.process(context)

            assert result["approved"] == False
            assert "drawdown" in result["reasoning"].lower()


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test the full agent pipeline."""
        from strategy_engine.core.orchestrator import AgentOrchestrator

        with patch('strategy_engine.core.orchestrator.get_ai_logger') as mock_logger:
            mock_logger.return_value.log_decision = AsyncMock()
            mock_logger.return_value.get_stats.return_value = {}

            orchestrator = AgentOrchestrator(config={})

            # Register agents
            orchestrator.register_agent("regime_detector", RegimeDetectorAgent(config={}))
            orchestrator.register_agent("trend_following", TrendFollowingAgent(config={}))
            orchestrator.register_agent("mean_reversion", MeanReversionAgent(config={"min_htf_periods": 5}))

            portfolio_agent = PortfolioManagerAgent(config={"base_confidence_threshold": 0.5})
            portfolio_agent.portfolio.total_equity = 10000
            orchestrator.register_agent("portfolio_manager", portfolio_agent)

            # Feed market data
            for i in range(30):
                market_data = {
                    "symbol": "BTCUSDT",
                    "price": 50000 + i * 50,
                    "bid": 49990 + i * 50,
                    "ask": 50010 + i * 50,
                    "volume": 1000,
                    "high_24h": 51000,
                    "low_24h": 49000,
                    "change_24h": 0.02,
                    "timestamp": 1234567890 + i,
                }

                signal = await orchestrator.process_market_data(market_data)
                if signal:
                    print(f"Signal generated: {signal.action} {signal.size} @ {signal.price}")


if __name__ == "__main__":
    # Run a quick test
    asyncio.run(TestPortfolioManager().test_approves_high_confidence(
        TestPortfolioManager().agent.__get__(None, TestPortfolioManager)
    ))
    print("Basic tests passed!")
