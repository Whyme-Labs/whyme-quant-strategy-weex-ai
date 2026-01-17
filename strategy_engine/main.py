"""Main entry point for the WEEX AI Strategy Engine."""

import asyncio
import signal
from typing import Optional
from loguru import logger

from shared.config import get_settings
from shared.discord import get_discord_notifier
from shared.llm import get_llm_analyzer
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
        self.discord = get_discord_notifier()
        self.llm = get_llm_analyzer()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_regime: dict = {}

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

        # Send startup notification to Discord
        await self.discord.send_status(
            "Engine Started",
            f"WhyMe Quant AI Strategy Engine initialized\n"
            f"**Symbol:** {self.settings.default_symbol}\n"
            f"**Max Leverage:** {self.settings.max_leverage}x\n"
            f"**LLM:** {self.settings.llm_model}",
            color=0x00FF00,
        )

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

        # Send shutdown notification
        await self.discord.send_status(
            "Engine Stopped",
            "WhyMe Quant AI Strategy Engine shutting down",
            color=0xFF6B6B,
        )

        # Stop components
        if self.orchestrator:
            await self.orchestrator.stop()

        if self.ai_uploader:
            # Upload any remaining AI logs
            await self.ai_uploader.force_upload_all()
            await self.ai_uploader.stop()

        if self.weex_client:
            await self.weex_client.close()

        # Close Discord and LLM clients
        await self.discord.close()
        await self.llm.close()

        logger.info("Strategy engine stopped")

    async def _main_loop(self):
        """Main trading loop."""
        symbol = self.settings.default_symbol
        logger.info(f"Starting main loop for {symbol}")
        iteration_count = 0
        last_regime_log = None  # Track last logged regime to avoid spam

        while self._running:
            try:
                iteration_count += 1

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

                # Get last regime from orchestrator
                if self.orchestrator.last_regime:
                    new_regime = self.orchestrator.last_regime
                    regime_data = new_regime.get("regime", {})
                    recommended = new_regime.get("recommended_strategy", "neutral")

                    # Log regime changes to Discord (not every tick)
                    regime_key = f"{regime_data.get('volatility')}-{regime_data.get('trend')}-{recommended}"
                    if regime_key != last_regime_log:
                        last_regime_log = regime_key
                        self._last_regime = regime_data

                        await self.discord.send_trace(
                            "Regime",
                            f"Market regime detected for {symbol}",
                            {
                                "Price": f"${market_data['price']:,.2f}",
                                "Volatility": regime_data.get("volatility", "unknown"),
                                "Trend": regime_data.get("trend", "unknown"),
                                "Volume": regime_data.get("volume", "unknown"),
                                "Strategy": recommended,
                                "Confidence": f"{new_regime.get('confidence', 0)*100:.0f}%",
                            }
                        )

                if signal:
                    logger.info(f"Trade approved by Portfolio Manager: {signal.action.value} {signal.size} {signal.symbol}")

                    # Update regime from signal metadata
                    if signal.metadata.get("regime"):
                        self._last_regime = signal.metadata["regime"]

                    # Log signal approval trace
                    await self.discord.send_trace(
                        "Portfolio",
                        f"Trade APPROVED by Portfolio Manager",
                        {
                            "Direction": signal.action.value.upper(),
                            "Size": f"{signal.size:.4f}",
                            "Entry": f"${signal.price or market_data['price']:,.2f}",
                            "TP": f"${signal.target_price:,.2f}" if signal.target_price else "Not set",
                            "SL": f"${signal.stop_price:,.2f}" if signal.stop_price else "Not set",
                            "Confidence": f"{signal.confidence*100:.0f}%",
                            "Strategy": signal.strategy,
                        }
                    )

                    # Execute the trade (Discord notification happens on success)
                    await self._execute_signal(signal, market_data)

                # Wait before next iteration
                await asyncio.sleep(self.settings.main_loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await self.discord.send_error(str(e), f"Main loop iteration {iteration_count}")
                await asyncio.sleep(self.settings.error_backoff_interval)

    async def _execute_signal(self, signal, market_data: dict):
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute
            market_data: Current market data
        """
        try:
            entry_price = signal.price or market_data["price"]

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

            # Get LLM analysis for the executed trade
            llm_analysis = await self.llm.analyze_signal(
                symbol=signal.symbol,
                price=entry_price,
                direction=signal.action.value,
                strategy=signal.strategy,
                confidence=signal.confidence,
                reasoning=signal.reason,
                regime=self._last_regime,
            )

            # Calculate risk/reward if TP and SL are set
            risk_reward = "N/A"
            if signal.stop_price and signal.target_price and entry_price:
                risk = abs(entry_price - signal.stop_price)
                reward = abs(signal.target_price - entry_price)
                if risk > 0:
                    risk_reward = f"{reward/risk:.2f}:1"

            # Send detailed execution notification to Discord
            await self.discord.send_signal(
                signal_type="TRADE EXECUTED",
                symbol=signal.symbol,
                direction=signal.action.value,
                price=entry_price,
                size=signal.size,
                confidence=signal.confidence,
                strategy=signal.strategy,
                reasoning=signal.reason,
                regime=self._last_regime,
                llm_analysis=llm_analysis,
            )

            # Send detailed trade info as follow-up
            tp_str = f"${signal.target_price:,.2f}" if signal.target_price else "Not set"
            sl_str = f"${signal.stop_price:,.2f}" if signal.stop_price else "Not set"

            await self.discord.send_status(
                "Trade Details",
                f"**Order ID:** `{order_id}`\n"
                f"**Entry Price:** ${entry_price:,.2f}\n"
                f"**Take Profit:** {tp_str}\n"
                f"**Stop Loss:** {sl_str}\n"
                f"**Risk/Reward:** {risk_reward}\n"
                f"**Position Size:** {signal.size}\n"
                f"**Confidence:** {signal.confidence*100:.0f}%\n"
                f"**Strategy:** {signal.strategy}",
                color=0x2ECC71 if signal.action.value == "buy" else 0xE74C3C,
            )

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
            await self.discord.send_error(
                str(e),
                f"Failed to execute {signal.action.value} {signal.symbol}",
            )


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
