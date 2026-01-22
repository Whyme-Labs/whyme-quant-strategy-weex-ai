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
from .services import (
    RedisClient,
    MarketDataService,
    IndicatorsService,
    PatternDetector,
    AlphaGenerator,
    TradeMemoryService,
    TradeJournal,
    # Statistical Edge Collection System
    EdgeRegistry,
    KellySizer,
    EdgeScanner,
    PerformanceTracker,
    # Support/Resistance Detection
    KeyLevelDetector,
    # Smart Money Concepts
    SMCDetector,
    # LLM Signal Validation
    LLMSignalValidator,
)
from .agents import (
    MarketAnalystAgent,
    RiskManagerAgent,
    ExecutorAgent,
    RegimeDetectorAgent,
    MeanReversionAgent,
    TrendFollowingAgent,
    TurtleTradingAgent,
    MomentumAgent,
    PivotAgent,
    PatternAgent,
    PortfolioManagerAgent,
    # Self-evolving RL agents
    ReflectionAgent,
    JudgeAgent,
    LearnerAgent,
)
from .models.memory import TradeStatus, ExitReason
from .models.edge import EdgeTradeAttribution
from .config.symbols import get_enabled_symbols, get_symbol_config, CORRELATION_LIMITS
from .config.edges import PREDEFINED_EDGES
from .loops import (
    PositionReviewLoop,
    TradeOutcomeLoop,
    ConsolidationLoop,
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

        # Redis client for cache persistence
        self.redis_client: Optional[RedisClient] = None

        # New services for multi-timeframe quant system
        self.market_data_service: Optional[MarketDataService] = None
        self.indicators_service: Optional[IndicatorsService] = None
        self.pattern_detector: Optional[PatternDetector] = None
        self.alpha_generator: Optional[AlphaGenerator] = None

        # Self-evolving RL system components
        self.trade_memory: Optional[TradeMemoryService] = None
        self.reflection_agent: Optional[ReflectionAgent] = None
        self.judge_agent: Optional[JudgeAgent] = None
        self.learner_agent: Optional[LearnerAgent] = None

        # Executor Agent - SINGLE POINT OF EXECUTION
        self.executor_agent: Optional[ExecutorAgent] = None

        # Learning loops
        self.position_review_loop: Optional[PositionReviewLoop] = None
        self.trade_outcome_loop: Optional[TradeOutcomeLoop] = None
        self.consolidation_loop: Optional[ConsolidationLoop] = None

        # Position monitoring task
        self._position_monitor_task: Optional[asyncio.Task] = None

        # Statistical Edge Collection System
        self.edge_registry: Optional[EdgeRegistry] = None
        self.kelly_sizer: Optional[KellySizer] = None
        self.edge_scanner: Optional[EdgeScanner] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self._edge_mode_enabled: bool = False  # Disable edge mode, use regime-based strategies (Turtle, Trend, MR)

        # Trade Journal for human-readable trade logging
        self.trade_journal: Optional[TradeJournal] = None

        # Smart Money Concepts Detector
        self.smc_detector: Optional[SMCDetector] = None

        # LLM Signal Validator (full override authority)
        self.llm_signal_validator: Optional[LLMSignalValidator] = None

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

        # Initialize Redis client for cache persistence
        self.redis_client = RedisClient(url=self.settings.redis_url)
        if await self.redis_client.connect():
            logger.info("Redis client connected for cache persistence")
        else:
            logger.warning("Redis unavailable - running with in-memory cache only")

        # Initialize multi-timeframe services
        self.market_data_service = MarketDataService(
            weex_client=self.weex_client,
            redis_client=self.redis_client,  # Pass Redis for persistence
            symbols=[self.settings.default_symbol],
            default_timeframes=["1h", "4h", "1d"],
        )
        await self.market_data_service.start()
        logger.info("Market Data Service initialized with 1H, 4H, 1D timeframes")

        self.indicators_service = IndicatorsService(
            market_data_service=self.market_data_service,
        )
        logger.info("Indicators Service initialized with pandas-ta")

        self.pattern_detector = PatternDetector(
            min_pattern_bars=20,
            peak_distance=5,
        )
        logger.info("Pattern Detector initialized")

        self.alpha_generator = AlphaGenerator(
            market_data_service=self.market_data_service,
            indicators_service=self.indicators_service,
            pattern_detector=self.pattern_detector,
        )
        logger.info("Alpha Generator initialized")

        # Initialize Key Level Detector (Support/Resistance)
        self.key_level_detector = KeyLevelDetector(
            market_data_service=self.market_data_service,
            redis_client=self.redis_client,
            swing_lookback=5,  # 5 candles for swing detection
            cluster_threshold=0.005,  # 0.5% for clustering
            volume_profile_bins=50,
        )
        logger.info("Key Level Detector initialized (S/R, Fibonacci, Volume Profile)")

        # Initialize SMC Detector (Smart Money Concepts)
        self.smc_detector = SMCDetector(
            market_data_service=self.market_data_service,
            redis_client=self.redis_client,
            key_level_detector=self.key_level_detector,
            swing_lookback=5,  # 5 candles for swing detection
            ob_lookback=20,  # 20 candles for order block detection
            fvg_lookback=50,  # 50 candles for FVG detection
            min_impulse_pct=0.015,  # 1.5% minimum impulse for OB
            min_fvg_pct=0.003,  # 0.3% minimum gap size
            equal_level_tolerance=0.002,  # 0.2% for equal highs/lows
        )
        logger.info("SMC Detector initialized (Order Blocks, FVG, BOS/CHoCH, Liquidity, Premium/Discount)")

        # Initialize LLM Signal Validator (full override authority)
        self.llm_signal_validator = LLMSignalValidator(
            llm_analyzer=self.llm,
            validation_cooldown=60,  # 60 second cooldown per symbol
            enable_override=True,  # Allow LLM to override rejected signals
        )
        logger.info("LLM Signal Validator initialized (full override authority enabled)")

        # Initialize Trade Memory Service (Triple Memory System)
        self.trade_memory = TradeMemoryService(redis_client=self.redis_client)
        await self.trade_memory.initialize()
        logger.info("Trade Memory Service initialized (Episodic + Semantic + Procedural)")

        # Initialize Statistical Edge Collection System
        await self._initialize_edge_system()

        # Initialize Trade Journal for human-readable logging
        self.trade_journal = TradeJournal(
            trade_memory=self.trade_memory,
            redis_client=self.redis_client,
            discord=self.discord,
        )
        logger.info("Trade Journal initialized (human-readable trade logging)")

        # Initialize Self-Evolving RL Agents
        self.reflection_agent = ReflectionAgent(
            config={
                "regime_change_sensitivity": 0.7,
                "min_profit_to_add": 0.01,
                "max_loss_before_review": -0.02,
                "max_hold_time_hours": 168,
            },
            llm_analyzer=self.llm,
        )

        self.judge_agent = JudgeAgent(
            config={
                "target_sharpe": 2.0,
                "min_return_for_good": 0.01,
                "max_loss_penalty": -0.05,
            },
            llm_analyzer=self.llm,
        )

        self.learner_agent = LearnerAgent(
            config={
                "min_trades_for_pattern": 5,
                "min_trades_for_evolution": 10,
                "max_change_per_cycle": 0.15,
                "confidence_for_evolution": 0.7,
                "lookback_days": 30,
            },
            trade_memory=self.trade_memory,
            llm_analyzer=self.llm,
        )
        logger.info("Self-evolving RL agents initialized (Reflection, Judge, Learner)")

        # Initialize Executor Agent - SINGLE POINT OF EXECUTION
        # This is the ONLY component that can execute trades
        self.executor_agent = ExecutorAgent(
            config={
                "slippage_tolerance": 0.001,
                "prefer_limit_orders": True,
            },
            weex_client=self.weex_client,
            trade_memory=self.trade_memory,
            discord=self.discord,
            llm_analyzer=self.llm,
            trade_journal=self.trade_journal,
        )
        logger.info("Executor Agent initialized (single point of execution, trade journal enabled)")

        # Initialize Learning Loops
        self.position_review_loop = PositionReviewLoop(
            weex_client=self.weex_client,
            reflection_agent=self.reflection_agent,
            trade_memory=self.trade_memory,
            executor_agent=self.executor_agent,  # Single point of execution
            regime_detector=None,  # Will be set after orchestrator init
            discord=self.discord,
            interval_seconds=3600,  # 1 hour
            symbol=self.settings.default_symbol,
        )

        self.trade_outcome_loop = TradeOutcomeLoop(
            trade_memory=self.trade_memory,
            judge_agent=self.judge_agent,
            llm_analyzer=self.llm,
            discord=self.discord,
            trade_journal=self.trade_journal,
        )

        self.consolidation_loop = ConsolidationLoop(
            trade_memory=self.trade_memory,
            learner_agent=self.learner_agent,
            discord=self.discord,
            trade_journal=self.trade_journal,
            lookback_days=30,
            enable_auto_evolution=True,
        )
        logger.info("Learning loops initialized (Position Review, Trade Outcome, Consolidation)")

        # Initialize AI log uploader
        self.ai_uploader = AILogUploader(
            weex_client=self.weex_client,
            upload_interval=self.settings.ai_log_upload_interval,
            batch_size=self.settings.ai_log_batch_size,
        )

        # Initialize orchestrator with agents
        self.orchestrator = AgentOrchestrator(
            config={},
            alpha_generator=self.alpha_generator,
            edge_scanner=self.edge_scanner,
            key_level_detector=self.key_level_detector,
            smc_detector=self.smc_detector,
            llm_validator=self.llm_signal_validator,  # LLM validation with full override
            discord=self.discord,  # Discord notifications for LLM decisions
        )

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
        # All strategy agents now use centralized IndicatorsService
        self.orchestrator.register_agent(
            "mean_reversion",
            MeanReversionAgent(
                config={
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "max_position_pct": 0.05,  # 5% max per trade
                    "target_return_pct": 0.02,  # 2% target
                    "timeframe": "4h",
                },
                indicators_service=self.indicators_service,
                market_data_service=self.market_data_service,
            ),
        )

        self.orchestrator.register_agent(
            "trend_following",
            TrendFollowingAgent(
                config={
                    "channel_period": 20,  # Turtle-style 20-day breakout
                    "ema_periods": [8, 20, 50],
                    "atr_period": 14,
                    "atr_multiplier": 2.0,  # Stop distance
                    "max_position_pct": 0.1,  # 10% max per trade
                    "vcp_min_candles": 7,  # VCP pattern minimum
                    "timeframe": "4h",
                },
                indicators_service=self.indicators_service,
                market_data_service=self.market_data_service,
            ),
        )

        # Turtle Trading Agent - Classic breakout system with REAL daily candles
        self.orchestrator.register_agent(
            "turtle_trading",
            TurtleTradingAgent(
                config={
                    "system1_entry": 20,    # 20-day breakout
                    "system1_exit": 10,     # 10-day exit
                    "system2_entry": 55,    # 55-day breakout
                    "system2_exit": 20,     # 20-day exit
                    "atr_period": 20,       # N calculation
                    "stop_atr_mult": 2.0,   # 2N stop
                    "max_units": 4,         # Max pyramid units
                    "risk_per_trade": 0.01, # 1% risk per unit
                },
                market_data_service=self.market_data_service,
                indicators_service=self.indicators_service,
            ),
        )

        # Momentum Agent - Trade with price momentum (ROC, RSI, MACD)
        self.orchestrator.register_agent(
            "momentum",
            MomentumAgent(
                config={
                    "roc_period": 14,        # Rate of change period
                    "roc_threshold": 0.05,   # 5% ROC threshold for signal
                    "rsi_period": 14,        # RSI period
                    "volume_mult": 2.0,      # Volume surge multiplier
                    "macd_fast": 12,         # MACD fast EMA
                    "macd_slow": 26,         # MACD slow EMA
                    "macd_signal": 9,        # MACD signal line
                    "max_position_pct": 0.08, # 8% max position
                    "atr_multiplier": 2.0,   # Stop distance in ATRs
                    "timeframe": "1h",       # Momentum on hourly
                },
                indicators_service=self.indicators_service,
                market_data_service=self.market_data_service,
            ),
        )

        # Pivot Agent - Trade pivot point support/resistance levels
        self.orchestrator.register_agent(
            "pivot",
            PivotAgent(
                config={
                    "bounce_threshold": 0.002,    # 0.2% for bounce detection
                    "breakout_threshold": 0.005,  # 0.5% for breakout confirmation
                    "max_position_pct": 0.06,     # 6% max position
                    "use_fibonacci": False,       # Use standard pivots
                    "atr_multiplier": 1.5,        # Stop distance
                    "timeframe": "1d",            # Pivot on daily
                },
                indicators_service=self.indicators_service,
                market_data_service=self.market_data_service,
            ),
        )

        # Pattern Agent - Trade classical chart patterns
        self.orchestrator.register_agent(
            "pattern",
            PatternAgent(
                config={
                    "min_confidence": 0.65,       # Minimum pattern confidence
                    "max_position_pct": 0.06,    # 6% max position
                    "timeframe": "4h",           # Pattern detection timeframe
                },
                market_data_service=self.market_data_service,
                pattern_detector=self.pattern_detector,
            ),
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
                "base_confidence_threshold": 0.40,  # Lower threshold for more trades
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

        # Note: ExecutorAgent is NOT registered with orchestrator
        # It's called directly by _execute_signal() as the single point of execution

        # Legacy: Keep market analyst for fallback
        self.orchestrator.register_agent(
            "market_analyst",
            MarketAnalystAgent(config={}),
        )

        logger.info("Strategy engine initialized successfully")

        # Get edge registry summary for startup message
        edge_summary = await self.edge_registry.get_registry_summary() if self.edge_registry else {}

        # Send startup notification to Discord
        await self.discord.send_status(
            "Engine Started",
            f"WhyMe Quant AI Strategy Engine initialized\n"
            f"**Symbol:** {self.settings.default_symbol}\n"
            f"**Max Leverage:** {self.settings.max_leverage}x\n"
            f"**LLM:** {self.settings.llm_model}\n\n"
            f"**Statistical Edge Collection System:**\n"
            f"- Mode: {'Edge-Based' if self._edge_mode_enabled else 'Regime-Based'}\n"
            f"- Registered Edges: {edge_summary.get('total_edges', 0)}\n"
            f"- Active Edges: {edge_summary.get('active_edges', 0)}\n"
            f"- Warming Edges: {edge_summary.get('warming_edges', 0)}\n"
            f"- Kelly Fraction: 25% (conservative)\n\n"
            f"**Multi-Timeframe System:**\n"
            f"- Timeframes: 1H, 4H, 1D (real candles)\n"
            f"- Strategies: Turtle, Trend, MR, Momentum, Pivot\n"
            f"- SMC: Order Blocks, FVG, BOS/CHoCH, Liquidity, Premium/Discount\n\n"
            f"**LLM Signal Validation:**\n"
            f"- Model: {self.settings.llm_model}\n"
            f"- Override Authority: FULL (can approve/reject any signal)\n"
            f"- Validation Cooldown: 60s per symbol\n\n"
            f"**Self-Evolving RL System:**\n"
            f"- Triple Memory: Episodic, Semantic, Procedural\n"
            f"- Position Monitor: Detects trade closes (60s interval)\n"
            f"- Position Review Loop: Hourly position reflection\n"
            f"- Trade Outcome Loop: Scoring & LLM reflection on close\n"
            f"- Consolidation Loop: Daily pattern extraction\n"
            f"- Trade Journal: Human-readable logging & daily summaries",
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

        # Start learning loops (run in background)
        await self.position_review_loop.start()
        await self.consolidation_loop.start()
        logger.info("Learning loops started (Position Review: hourly, Consolidation: daily)")

        # Start position monitor for trade outcome detection
        self._position_monitor_task = asyncio.create_task(self._monitor_positions())
        logger.info("Position monitor started (active only when positions are open)")

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

        # Stop position monitor
        if self._position_monitor_task:
            self._position_monitor_task.cancel()
            try:
                await self._position_monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("Position monitor stopped")

        # Stop learning loops
        if self.position_review_loop:
            await self.position_review_loop.stop()

        if self.consolidation_loop:
            await self.consolidation_loop.stop()

        logger.info("Learning loops stopped")

        # Stop components
        if self.market_data_service:
            await self.market_data_service.stop()

        if self.orchestrator:
            await self.orchestrator.stop()

        if self.ai_uploader:
            # Upload any remaining AI logs
            await self.ai_uploader.force_upload_all()
            await self.ai_uploader.stop()

        if self.weex_client:
            await self.weex_client.close()

        # Close Redis client
        if self.redis_client:
            await self.redis_client.close()

        # Close Discord and LLM clients
        await self.discord.close()
        await self.llm.close()

        logger.info("Strategy engine stopped")

    async def _main_loop(self):
        """Main trading loop.

        Routes to edge-based or regime-based loop based on configuration.
        Now iterates over multiple symbols for more trading opportunities.
        """
        # Use edge-based loop if enabled and initialized
        if self._edge_mode_enabled and self.edge_scanner:
            await self._edge_based_main_loop()
            return

        # Get list of trading symbols
        symbols = [s.strip() for s in self.settings.trading_symbols.split(",")]
        logger.info(f"Starting multi-symbol main loop for {len(symbols)} symbols: {symbols}")
        iteration_count = 0
        last_regime_log = {}  # Track last logged regime per symbol to avoid spam

        while self._running:
            try:
                iteration_count += 1

                # Clear stale analysis at start of each cycle
                self.orchestrator.last_strategy_analysis.clear()

                # Process each symbol
                for symbol in symbols:
                    if not self._running:
                        break

                    try:
                        # Fetch market data (ticker + OHLCV candles for proper calculations)
                        ticker = await self.weex_client.get_ticker(symbol)

                        # Get OHLCV candle data from MarketDataService for proper ATR, channel calculations
                        candles_1h_df = await self.market_data_service.get_candles(symbol, "1h", limit=50)
                        candles_4h_df = await self.market_data_service.get_candles(symbol, "4h", limit=50)
                        candles_1d_df = await self.market_data_service.get_candles(symbol, "1d", limit=60)  # For Turtle Trading (55-day)

                        # Convert DataFrames to list of dicts for agent consumption
                        candles_1h = candles_1h_df.to_dict('records') if not candles_1h_df.empty else []
                        candles_4h = candles_4h_df.to_dict('records') if not candles_4h_df.empty else []
                        candles_1d = candles_1d_df.to_dict('records') if not candles_1d_df.empty else []

                        # Fetch account info for portfolio manager (including positions for restart awareness)
                        account_info = {}
                        try:
                            assets = await self.weex_client.get_assets()
                            if assets and isinstance(assets, list):
                                usdt_asset = next((a for a in assets if a.get("coinName") == "USDT"), None)
                                if usdt_asset:
                                    account_info = {
                                        "equity": float(usdt_asset.get("equity", 0)),
                                        "available": float(usdt_asset.get("available", 0)),
                                        "usedMargin": float(usdt_asset.get("frozen", 0)),
                                        "unrealizedPnl": float(usdt_asset.get("unrealizePnl", 0)),
                                        "positions": [],  # Will be populated below
                                    }

                            # Fetch open positions to ensure portfolio awareness across restarts
                            positions = await self.weex_client.get_positions(symbol)
                            if positions and isinstance(positions, list):
                                account_info["positions"] = [
                                    {
                                        "symbol": p.get("symbol", symbol),
                                        "side": p.get("side", "LONG").lower(),  # API returns 'LONG'/'SHORT'
                                        "size": float(p.get("size", 0)),  # API returns 'size' not 'total'
                                        "entryPrice": float(p.get("averageOpenPrice", 0)),
                                        "unrealizedPnl": float(p.get("unrealizePnl", 0)),
                                        "margin": float(p.get("marginSize", 0)),
                                    }
                                    for p in positions
                                    if float(p.get("size", 0)) > 0
                                ]
                                if account_info["positions"]:
                                    logger.debug(f"{symbol}: {len(account_info['positions'])} open positions")
                        except Exception as e:
                            logger.debug(f"Failed to fetch account info for {symbol}: {e}")

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
                            # OHLCV candle data for proper indicator calculations
                            "candles_1h": candles_1h,
                            "candles_4h": candles_4h,
                            "candles_1d": candles_1d,
                            # Account info for portfolio manager
                            "account_info": account_info,
                        }

                        # Process through agent orchestrator
                        signal = await self.orchestrator.process_market_data(market_data)

                        # Get last regime from orchestrator
                        if self.orchestrator.last_regime:
                            new_regime = self.orchestrator.last_regime
                            regime_data = new_regime.get("regime", {})
                            recommended = new_regime.get("recommended_strategy", "neutral")

                            # Log regime changes to Discord (per-symbol, not every tick)
                            regime_key = f"{symbol}-{regime_data.get('volatility')}-{regime_data.get('trend')}-{recommended}"
                            if regime_key != last_regime_log.get(symbol):
                                last_regime_log[symbol] = regime_key
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
                            logger.info(f"Trade approved for {symbol}: {signal.action.value} size={signal.size or 'N/A'}")

                            # Update regime from signal metadata
                            if signal.metadata.get("regime"):
                                self._last_regime = signal.metadata["regime"]

                            # Log signal approval trace
                            entry_price = signal.price or market_data.get('price', 0)
                            await self.discord.send_trace(
                                "Portfolio",
                                f"Trade APPROVED for {symbol}",
                                {
                                    "Direction": signal.action.value.upper(),
                                    "Size": f"{signal.size:.4f}" if signal.size else "N/A",
                                    "Entry": f"${entry_price:,.2f}",
                                    "TP": f"${signal.target_price:,.2f}" if signal.target_price else "Not set",
                                    "SL": f"${signal.stop_price:,.2f}" if signal.stop_price else "Not set",
                                    "Confidence": f"{signal.confidence*100:.0f}%" if signal.confidence else "N/A",
                                    "Strategy": signal.strategy or "unknown",
                                }
                            )

                            # Execute the trade (Discord notification happens on success)
                            await self._execute_signal(signal, market_data)

                    except Exception as symbol_error:
                        logger.warning(f"Error processing {symbol}: {symbol_error}")
                        continue  # Continue to next symbol

                # Periodic strategy analysis log (every 60 iterations = ~5 minutes)
                if iteration_count % 60 == 0 and self.orchestrator.last_strategy_analysis:
                    # Build per-symbol analysis with clear labeling
                    for symbol, analyses in self.orchestrator.last_strategy_analysis.items():
                        # Only log symbols with signals
                        signals_found = [a for a in analyses if a[1]]  # a[1] = has_signal
                        if signals_found:
                            signal_fields = {}
                            for strategy_name, has_signal, reasoning in analyses:
                                if has_signal:
                                    # Show full reasoning (no truncation)
                                    signal_fields[f"✅ {strategy_name}"] = reasoning

                            await self.discord.send_trace(
                                f"📊 {symbol}",
                                f"{len(signals_found)} signal(s) detected",
                                signal_fields
                            )

                    # Also send a summary of pairs with no signals
                    no_signal_pairs = [
                        sym for sym, analyses in self.orchestrator.last_strategy_analysis.items()
                        if not any(a[1] for a in analyses)
                    ]
                    if no_signal_pairs:
                        await self.discord.send_trace(
                            "Strategy Summary",
                            f"Scanned {len(symbols)} pairs",
                            {
                                "🔄 Waiting": ", ".join(no_signal_pairs),
                                "Active Signals": str(len(symbols) - len(no_signal_pairs)),
                            }
                        )

                # Wait before next iteration (after processing all symbols)
                await asyncio.sleep(self.settings.main_loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await self.discord.send_error(str(e), f"Main loop iteration {iteration_count}")
                await asyncio.sleep(self.settings.error_backoff_interval)

    async def _execute_signal(self, signal, market_data: dict):
        """Execute a trading signal via ExecutorAgent.

        Args:
            signal: Trading signal to execute
            market_data: Current market data
        """
        try:
            # Execute via ExecutorAgent (single point of execution)
            # All Discord logging, trade recording, and LLM analysis
            # is handled internally by the ExecutorAgent
            result = await self.executor_agent.execute_open(
                signal=signal,
                market_data=market_data,
                regime=self._last_regime.copy() if self._last_regime else {},
            )

            if result and result.get("success"):
                order_id = result.get("order_id")
                trade_id = result.get("trade_id")
                logger.info(f"Trade executed via ExecutorAgent: order={order_id}, trade={trade_id}")

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
            else:
                logger.warning("ExecutorAgent failed to execute signal")

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            await self.discord.send_error(
                str(e),
                f"Failed to execute {signal.action.value} {signal.symbol}",
            )

    async def _monitor_positions(self):
        """Monitor positions to detect trade closes.

        Only actively monitors when there are open positions to track.
        Uses ExecutorAgent's order_to_trade mapping for tracking.
        When detected, calls trade_outcome_loop.on_trade_closed().
        """
        logger.info("Position monitor started")
        symbol = self.settings.default_symbol

        while self._running:
            try:
                # Get order tracking from ExecutorAgent
                order_to_trade = self.executor_agent.get_order_to_trade_mapping()

                # Only check frequently when there are trades to monitor
                if not order_to_trade:
                    await asyncio.sleep(300)  # Sleep 5 minutes when no positions
                    continue

                await asyncio.sleep(60)  # Check every 60 seconds when monitoring

                # Get current positions
                positions = await self.weex_client.get_positions(symbol)

                # Get current ticker for exit price
                ticker = await self.weex_client.get_ticker(symbol)
                current_price = float(ticker.get("last", 0))

                # Get current regime
                current_regime = self._last_regime.copy() if self._last_regime else {}

                # Find position IDs that are still open
                open_position_ids = set()
                if isinstance(positions, list):
                    for p in positions:
                        size = float(p.get("size", 0))  # API returns 'size' not 'total'
                        if size != 0:
                            # Position is still open
                            open_position_ids.add(str(p.get("id", "")))

                # Check each tracked trade
                closed_trades = []
                for order_id, trade_id in list(order_to_trade.items()):
                    # Get the trade record to check its status
                    trade = await self.trade_memory.get_trade(trade_id)
                    if not trade:
                        # Trade not found, remove from tracking
                        self.executor_agent.remove_order_mapping(order_id)
                        continue

                    if trade.status != TradeStatus.OPEN:
                        # Already processed, remove from tracking
                        self.executor_agent.remove_order_mapping(order_id)
                        continue

                    # Check if this position is still open
                    # We check by order ID or by looking at open positions
                    position_closed = True
                    for p in (positions if isinstance(positions, list) else []):
                        # Check if position matches this trade
                        if abs(float(p.get("size", 0))) > 0:  # API returns 'size' not 'total'
                            # There's still an open position, assume trade is still active
                            position_closed = False
                            break

                    if position_closed:
                        closed_trades.append((order_id, trade_id))

                # Process closed trades
                for order_id, trade_id in closed_trades:
                    logger.info(f"Detected closed position for trade {trade_id}")

                    # Determine exit reason - check order history
                    exit_reason = ExitReason.MANUAL  # Default
                    try:
                        order_history = await self.weex_client.get_order_history(symbol, limit=10)
                        if isinstance(order_history, list):
                            for order in order_history:
                                if str(order.get("orderId")) == order_id:
                                    # Check order details for stop/tp hits
                                    order_type = order.get("orderType", "").lower()
                                    if "stop" in order_type:
                                        exit_reason = ExitReason.STOP_LOSS
                                    elif "profit" in order_type or "tp" in order_type:
                                        exit_reason = ExitReason.TAKE_PROFIT
                                    break
                    except Exception as e:
                        logger.warning(f"Could not determine exit reason: {e}")

                    # Call trade outcome loop
                    await self.trade_outcome_loop.on_trade_closed(
                        trade_id=trade_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        exit_regime=current_regime,
                    )

                    # Remove from tracking via ExecutorAgent
                    self.executor_agent.remove_order_mapping(order_id)

                    # Note: Discord notification is handled by TradeOutcomeLoop

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                # Don't spam errors, wait longer before retry
                await asyncio.sleep(30)

        logger.info("Position monitor stopped")

    # =========================================================================
    # STATISTICAL EDGE COLLECTION SYSTEM
    # =========================================================================

    async def _initialize_edge_system(self):
        """Initialize the Statistical Edge Collection System."""
        logger.info("Initializing Statistical Edge Collection System...")

        # Initialize Edge Registry
        self.edge_registry = EdgeRegistry(redis_client=self.redis_client)
        await self.edge_registry.initialize()

        # Register predefined edges
        for edge in PREDEFINED_EDGES:
            await self.edge_registry.register_edge(edge)

        logger.info(f"Registered {len(PREDEFINED_EDGES)} predefined edges")

        # Initialize Kelly Sizer
        self.kelly_sizer = KellySizer(
            kelly_fraction=0.25,  # Use 25% of full Kelly (conservative)
            max_position_pct=0.10,  # Max 10% per trade
            min_position_pct=0.01,  # Min 1% to be worth trading
            max_total_exposure=CORRELATION_LIMITS.get("total", 0.50),
        )
        logger.info("Kelly Sizer initialized (25% fractional Kelly)")

        # Initialize Edge Scanner
        self.edge_scanner = EdgeScanner(
            edge_registry=self.edge_registry,
            market_data=self.market_data_service,
            indicators=self.indicators_service,
        )
        logger.info("Edge Scanner initialized")

        # Initialize Performance Tracker
        self.performance_tracker = PerformanceTracker(
            edge_registry=self.edge_registry,
            redis_client=self.redis_client,
            discord=self.discord,
        )
        logger.info("Performance Tracker initialized")

        # Get registry summary
        summary = await self.edge_registry.get_registry_summary()
        logger.info(
            f"Edge Collection System ready: "
            f"{summary['total_edges']} edges, "
            f"{summary['active_edges']} active, "
            f"{summary['warming_edges']} warming"
        )

    async def _edge_based_main_loop(self):
        """Edge-based main trading loop.

        Implements the Statistical Edge Collection philosophy:
        "Systematically collecting probability advantages in local market inefficiencies."
        """
        logger.info("Starting edge-based main loop")
        iteration_count = 0

        # Get enabled symbols
        enabled_symbols = get_enabled_symbols()
        symbol_list = [s.symbol for s in enabled_symbols]
        logger.info(f"Trading symbols: {symbol_list}")

        while self._running:
            try:
                iteration_count += 1

                # 1. Scan all edges across all symbols
                edge_signals = await self.edge_scanner.scan_all_edges()

                if not edge_signals:
                    await asyncio.sleep(self.settings.main_loop_interval)
                    continue

                logger.info(f"Edge scan found {len(edge_signals)} signal(s)")

                # 2. Get current account equity
                account_equity = await self._get_account_equity()
                if account_equity <= 0:
                    logger.warning("Could not get account equity")
                    await asyncio.sleep(self.settings.main_loop_interval)
                    continue

                # 3. Get current prices and volatilities for all symbols
                prices = {}
                volatilities = {}
                for symbol in symbol_list:
                    try:
                        ticker = await self.weex_client.get_ticker(symbol)
                        prices[symbol] = float(ticker.get("last", 0))

                        # Get ATR for volatility
                        indicators = await self.indicators_service.calculate_indicators(
                            symbol, "1d"
                        )
                        if indicators and "atr" in indicators:
                            atr_pct = indicators["atr"] / prices[symbol] if prices[symbol] > 0 else 0.03
                            volatilities[symbol] = atr_pct
                    except Exception as e:
                        logger.warning(f"Could not get data for {symbol}: {e}")

                # 4. Calculate position sizes for all signals
                position_sizes = {}
                current_exposure = 0.0

                for signal in edge_signals:
                    # Check edge health
                    health = await self.edge_registry.check_edge_health(signal.edge_id)
                    if health and health.should_pause:
                        logger.debug(f"Skipping edge {signal.edge_id}: {health.status_reason}")
                        continue

                    # Calculate Kelly-based position size
                    price = prices.get(signal.symbol, signal.price)
                    vol = volatilities.get(signal.symbol, 0.03)

                    vol_adj = self.kelly_sizer.get_volatility_adjustment(vol)

                    position = self.kelly_sizer.calculate_position_size(
                        edge=signal.edge,
                        account_equity=account_equity,
                        current_price=price,
                        volatility_adjustment=vol_adj,
                        current_exposure=current_exposure,
                    )

                    if position.can_trade:
                        position_sizes[signal.signal_id] = {
                            "signal": signal,
                            "position": position,
                            "price": price,
                        }
                        current_exposure += position.position_pct

                # 5. Execute trades for signals with valid position sizes
                for signal_id, data in position_sizes.items():
                    signal = data["signal"]
                    position = data["position"]
                    price = data["price"]

                    # Convert EdgeSignal to a format ExecutorAgent can use
                    await self._execute_edge_signal(signal, position, price)

                # 6. Log edge scan summary
                if position_sizes:
                    logger.info(
                        f"Executed {len(position_sizes)} edge signal(s), "
                        f"total exposure: {current_exposure:.1%}"
                    )

                # Wait before next iteration
                await asyncio.sleep(self.settings.main_loop_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in edge-based main loop: {e}")
                await self.discord.send_error(str(e), f"Edge loop iteration {iteration_count}")
                await asyncio.sleep(self.settings.error_backoff_interval)

    async def _execute_edge_signal(self, signal, position, current_price: float):
        """Execute an edge signal via ExecutorAgent.

        Args:
            signal: EdgeSignal object
            position: PositionSize object
            current_price: Current price
        """
        try:
            # Build market data dict
            market_data = {
                "symbol": signal.symbol,
                "price": current_price,
                "bid": signal.market_data.get("bid", current_price * 0.9999),
                "ask": signal.market_data.get("ask", current_price * 1.0001),
                "volume": signal.market_data.get("volume", 0),
            }

            # Create a Signal-like object for ExecutorAgent
            from .core.base import Signal, SignalAction

            action = SignalAction.BUY if signal.side == "long" else SignalAction.SELL

            legacy_signal = Signal(
                action=action,
                symbol=signal.symbol,
                price=signal.suggested_entry,
                stop_price=signal.suggested_stop,
                target_price=signal.suggested_take_profit,
                size=position.size,
                confidence=signal.edge.expectancy if signal.edge else 0.5,
                strategy=f"edge:{signal.edge_id}",
                reasoning=f"Edge signal: {signal.edge.name if signal.edge else signal.edge_id}",
                metadata={
                    "edge_id": signal.edge_id,
                    "kelly_raw": position.kelly_raw,
                    "kelly_fractional": position.kelly_fractional,
                    "edge_expectancy": position.edge_expectancy,
                    "edge_win_rate": position.edge_win_rate,
                    "position_pct": position.position_pct,
                },
            )

            # Execute via ExecutorAgent
            result = await self.executor_agent.execute_open(
                signal=legacy_signal,
                market_data=market_data,
                regime=self._last_regime.copy() if self._last_regime else {},
            )

            if result and result.get("success"):
                trade_id = result.get("trade_id")
                order_id = result.get("order_id")

                # Record attribution for performance tracking
                await self.performance_tracker.record_entry(
                    edge_id=signal.edge_id,
                    trade_id=trade_id,
                    position_size=position.size,
                    entry_price=current_price,
                    kelly_used=position.kelly_fractional,
                )

                logger.info(
                    f"Edge signal executed: {signal.edge_id} "
                    f"(trade={trade_id}, size={position.size:.4f}, "
                    f"kelly={position.kelly_fractional:.2%})"
                )

                # Upload AI logs
                if order_id:
                    ai_logger = get_ai_logger()
                    pending = await ai_logger.get_pending_uploads()
                    for decision in pending:
                        if decision.order_id is None:
                            decision.order_id = int(order_id)
                    await self.ai_uploader.upload_for_order(int(order_id))
            else:
                logger.warning(f"Failed to execute edge signal {signal.edge_id}")

        except Exception as e:
            logger.error(f"Failed to execute edge signal: {e}")
            await self.discord.send_error(
                str(e),
                f"Failed to execute edge {signal.edge_id}",
            )

    async def _get_account_equity(self) -> float:
        """Get current account equity.

        Returns:
            Account equity in USD
        """
        try:
            account = await self.weex_client.get_account()
            if isinstance(account, dict):
                # Try different field names
                equity = account.get("equity") or account.get("totalEquity") or account.get("available")
                if equity:
                    return float(equity)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
            return 0.0

    async def _run_edge_health_check(self):
        """Run periodic edge health check."""
        try:
            health_reports = await self.performance_tracker.check_all_edge_health()

            # Log summary
            healthy = sum(1 for h in health_reports if h.status.value == "healthy")
            degrading = sum(1 for h in health_reports if h.status.value == "degrading")
            broken = sum(1 for h in health_reports if h.status.value == "broken")

            if degrading > 0 or broken > 0:
                logger.warning(
                    f"Edge health check: {healthy} healthy, "
                    f"{degrading} degrading, {broken} broken"
                )
            else:
                logger.info(f"Edge health check: {healthy} healthy edges")

        except Exception as e:
            logger.error(f"Edge health check failed: {e}")


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
