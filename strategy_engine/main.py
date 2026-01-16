"""Main entry point for the WEEX AI Strategy Engine."""

import asyncio
import signal
from typing import Optional
from loguru import logger

from shared.config import get_settings
from weex_client import WeexClient
from ai_logging import AILogUploader, get_ai_logger
from .core.orchestrator import AgentOrchestrator
from .agents import (
    MarketAnalystAgent,
    RiskManagerAgent,
    ExecutionAgent,
    RegimeDetectorAgent,
    MeanReversionAgent,
    TrendFollowingAgent,
    PortfolioManagerAgent,
)


class StrategyEngine:
    """Main strategy engine for WEEX AI trading."""

    def __init__(self):
        """Initialize the strategy engine."""
        self.settings = get_settings()
        self.weex_client: Optional[WeexClient] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.ai_uploader: Optional[AILogUploader] = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing WEEX AI Strategy Engine...")

        # Initialize WEEX client
        self.weex_client = WeexClient(
            api_key=self.settings.weex_api_key,
            secret_key=self.settings.weex_secret_key,
            passphrase=self.settings.weex_passphrase,
            base_url=self.settings.weex_api_url,
        )

        # Test connection
        if not await self.weex_client.test_connection():
            raise Exception("Failed to connect to WEEX API")

        # Initialize AI log uploader
        self.ai_uploader = AILogUploader(
            weex_client=self.weex_client,
            upload_interval=self.settings.ai_log_upload_interval,
            batch_size=self.settings.ai_log_batch_size,
        )

        # Initialize orchestrator with agents
        self.orchestrator = AgentOrchestrator(config={})

        # Register agents - Regime-based multi-agent architecture
        # Based on research: "Some regimes reward trend following. Others reward mean reversion."

        # Stage 1: Regime Detection
        self.orchestrator.register_agent(
            "regime_detector",
            RegimeDetectorAgent(config={
                "ema_periods": [8, 20, 50],
                "atr_period": 14,
                "lookback_period": 50,
            }),
        )

        # Stage 2: Strategy Agents (routed by regime)
        self.orchestrator.register_agent(
            "mean_reversion",
            MeanReversionAgent(config={
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "max_position_pct": 0.05,  # 5% max per trade
                "target_return_pct": 0.02,  # 2% target
            }),
        )

        self.orchestrator.register_agent(
            "trend_following",
            TrendFollowingAgent(config={
                "channel_period": 20,  # Turtle-style 20-day breakout
                "ema_periods": [8, 20, 50],
                "atr_period": 14,
                "atr_multiplier": 2.0,  # Stop distance
                "max_position_pct": 0.1,  # 10% max per trade
                "vcp_min_candles": 7,  # VCP pattern minimum
            }),
        )

        # Stage 3: Portfolio Management (the execution gatekeeper)
        # This is the critical bridge between signals and execution
        # Dynamic confidence threshold based on portfolio state and market conditions
        self.orchestrator.register_agent(
            "portfolio_manager",
            PortfolioManagerAgent(config={
                "max_portfolio_exposure": 0.5,  # Max 50% of equity exposed
                "max_single_position": 0.1,  # Max 10% per position
                "max_correlated_exposure": 0.3,  # Max 30% in correlated assets
                "base_confidence_threshold": 0.6,  # Starting confidence threshold
                "max_daily_drawdown": 0.05,  # Stop trading at 5% daily drawdown
                "win_rate_lookback": 20,  # Calculate win rate from last 20 trades
            }),
        )

        # Stage 4: Risk Management (additional checks)
        self.orchestrator.register_agent(
            "risk_manager",
            RiskManagerAgent(config={
                "max_leverage": self.settings.max_leverage,
                "max_position_pct": 0.1,
            }),
        )

        # Stage 4: Execution
        self.orchestrator.register_agent(
            "execution_agent",
            ExecutionAgent(config={}),
        )

        # Legacy: Keep market analyst for fallback
        self.orchestrator.register_agent(
            "market_analyst",
            MarketAnalystAgent(config={}),
        )

        logger.info("Strategy engine initialized successfully")

    async def start(self):
        """Start the strategy engine."""
        if self._running:
            logger.warning("Engine already running")
            return

        self._running = True
        logger.info("Starting WEEX AI Strategy Engine...")

        # Start components
        await self.orchestrator.start()
        await self.ai_uploader.start()

        # Start main loop
        await self._main_loop()

    async def stop(self):
        """Stop the strategy engine."""
        logger.info("Stopping WEEX AI Strategy Engine...")
        self._running = False
        self._shutdown_event.set()

        # Stop components
        if self.orchestrator:
            await self.orchestrator.stop()

        if self.ai_uploader:
            # Upload any remaining AI logs
            await self.ai_uploader.force_upload_all()
            await self.ai_uploader.stop()

        if self.weex_client:
            await self.weex_client.close()

        logger.info("Strategy engine stopped")

    async def _main_loop(self):
        """Main trading loop."""
        symbol = self.settings.default_symbol
        logger.info(f"Starting main loop for {symbol}")

        while self._running:
            try:
                # Fetch market data
                ticker = await self.weex_client.get_ticker(symbol)

                market_data = {
                    "symbol": symbol,
                    "price": float(ticker.get("last", 0)),
                    "bid": float(ticker.get("bestBid", 0)),
                    "ask": float(ticker.get("bestAsk", 0)),
                    "volume": float(ticker.get("baseVolume", 0)),
                    "high_24h": float(ticker.get("high24h", 0)),
                    "low_24h": float(ticker.get("low24h", 0)),
                    "change_24h": float(ticker.get("change24h", 0)),
                    "timestamp": ticker.get("timestamp"),
                }

                # Process through agent orchestrator
                signal = await self.orchestrator.process_market_data(market_data)

                if signal:
                    logger.info(f"Signal generated: {signal.action} {signal.size} {signal.symbol}")

                    # Execute the trade
                    await self._execute_signal(signal)

                # Wait before next iteration
                await asyncio.sleep(5)  # Poll every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)  # Back off on error

    async def _execute_signal(self, signal):
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute
        """
        try:
            # Place order
            result = await self.weex_client.place_order(
                symbol=signal.symbol,
                side=signal.action.value,
                order_type="limit" if signal.price else "market",
                size=str(signal.size),
                price=str(signal.price) if signal.price else None,
            )

            order_id = result.get("orderId")
            logger.info(f"Order placed: {order_id}")

            # Upload AI logs for this order
            if order_id:
                ai_logger = get_ai_logger()
                # Update pending decisions with order_id
                pending = await ai_logger.get_pending_uploads()
                for decision in pending:
                    if decision.order_id is None:
                        decision.order_id = int(order_id)

                # Upload immediately
                await self.ai_uploader.upload_for_order(int(order_id))

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")


async def main():
    """Main entry point."""
    engine = StrategyEngine()

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(engine.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        await engine.initialize()
        await engine.start()
    except Exception as e:
        logger.error(f"Engine error: {e}")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
