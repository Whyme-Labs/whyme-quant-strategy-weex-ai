# WhyMe Quant Strategy - WEEX AI

A multi-agent AI trading strategy for the **WEEX AI Hackathon: AI Wars Alpha Awakens**.

## Hackathon Details

- **Event**: WEEX AI Hackathon Preliminary Round (Forked Entry)
- **Prize Pool**: $1,880,000 USDT
- **Registration Deadline**: January 18, 2026, 23:59 (UTC+8)
- **Preliminary Round**: January 19 - February 2, 2026
- **Competition Fund**: 1,000 USDT allocated per BUIDL

## Architecture

Based on research insights: *"There are only 2 trading strategies in the world: Mean Reversion and Trend Following. Some regimes reward trend following. Others reward mean reversion. Running both smooths returns and reduces drawdowns."*

Our system features a **Self-Evolving Agentic RL Architecture** that learns from every trade and autonomously evolves its parameters within safe bounds.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Self-Evolving Agentic Trading System                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    Triple Memory System (Redis)                     │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │     │
│  │  │   EPISODIC   │  │   SEMANTIC   │  │      PROCEDURAL        │   │     │
│  │  │   (Trades)   │  │  (Patterns)  │  │   (Strategy Params)    │   │     │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│  ┌─────────────────────────────────┼─────────────────────────────────┐      │
│  │                         Learning Loops                             │      │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │      │
│  │  │ POSITION REVIEW │  │  TRADE OUTCOME  │  │  CONSOLIDATION  │   │      │
│  │  │   (Hourly)      │  │  (On Close)     │  │    (Daily)      │   │      │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │      │
│  │           └────────────────────┴────────────────────┘             │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                   │                                          │
│  ┌────────────────────────────────┼───────────────────────────────────┐     │
│  │                    Reflection Agents                                │     │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐  │     │
│  │  │ REFLECTION    │  │    JUDGE      │  │       LEARNER         │  │     │
│  │  │ (Position Mgr)│  │ (Trade Score) │  │ (Pattern Extraction)  │  │     │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘  │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Existing Trading Pipeline                         │    │
│  │  RegimeDetector → Strategies → PortfolioMgr → RiskMgr → Execution   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Statistical Edge Collection System

Based on the insight: *"Quantification is not about predicting future prices, but systematically collecting probability advantages in local market inefficiencies."*

### Core Principles

1. **Every signal is an "edge"** with measurable statistical properties
2. **Position size based on edge magnitude** using Kelly Criterion
3. **Law of Large Numbers** - high frequency dilutes single-trade noise
4. **Diversification** across symbols, timeframes, and edge types
5. **Track convergence** to expected value

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Statistical Edge Collection System                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Edge Registry (Redis)                          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │  edge:mr_rsi_ob │  │ edge:tf_breakout│  │  edge:turtle_s1     │  │   │
│  │  │  win_rate: 0.52 │  │ win_rate: 0.38  │  │  win_rate: 0.35     │  │   │
│  │  │  payoff: 1.8:1  │  │ payoff: 3.2:1   │  │  payoff: 4.5:1      │  │   │
│  │  │  edge: +0.12    │  │ edge: +0.22     │  │  edge: +0.18        │  │   │
│  │  │  kelly: 6.7%    │  │ kelly: 6.9%     │  │  kelly: 4.0%        │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │                    Edge-Based Trading Pipeline                         │  │
│  │                                                                        │  │
│  │   Market Data → Edge Scanner → Kelly Sizer → Diversifier → Executor   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Performance Attribution                            │   │
│  │                                                                        │   │
│  │   Track: actual_pnl vs expected_pnl → convergence to edge             │   │
│  │   Alert: when rolling performance < 50% of expected                   │   │
│  │   Auto-disable: edges with negative rolling expectancy                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Edge Services

| Service | Purpose | Key Features |
|---------|---------|--------------|
| **EdgeRegistry** | Stores and tracks all edges | Win rate, payoff ratio, expectancy, Kelly |
| **KellySizer** | Position sizing | 25% fractional Kelly, volatility adjustment |
| **EdgeScanner** | Scans for edge signals | Multi-symbol, multi-timeframe scanning |
| **PerformanceTracker** | Attribution & convergence | Expected vs actual P&L tracking |

### Kelly Criterion Sizing

```python
# Edge formula (expectancy per trade)
expectancy = (win_rate × avg_win) - (loss_rate × avg_loss)

# Kelly Criterion (optimal fraction)
kelly = (win_rate × payoff - loss_rate) / payoff

# Fractional Kelly (25% for safety)
position_size = kelly × 0.25
```

### Multi-Symbol Support

| Symbol | Max Position | Correlation Group |
|--------|-------------|-------------------|
| BTCUSDT | 15% | btc |
| ETHUSDT | 10% | eth |

### Predefined Edges

| Edge ID | Type | Timeframe | Entry Conditions |
|---------|------|-----------|------------------|
| `mr_rsi_oversold_btc_4h` | Mean Reversion | 4H | RSI < 30, Below BB Lower |
| `mr_rsi_overbought_btc_4h` | Mean Reversion | 4H | RSI > 70, Above BB Upper |
| `tf_channel_breakout_btc_1d` | Trend Following | 1D | 20-day channel breakout |
| `tf_vcp_btc_4h` | Trend Following | 4H | VCP pattern + ATR contraction |
| `turtle_s1_btc_1d` | Turtle | 1D | 20-day breakout (System 1) |
| `turtle_s2_btc_1d` | Turtle | 1D | 55-day breakout (System 2) |

---

## Strategy: Regime-Based Multi-Agent Trading

This strategy uses specialized AI agents that collaborate based on detected market regimes:

### Services Layer

| Service | Purpose | Features |
|---------|---------|----------|
| **MarketDataService** | Multi-timeframe OHLCV data | 1H, 4H, 1D candles with Redis persistence |
| **IndicatorsService** | Technical indicator calculations | EMA, RSI, Bollinger Bands, ATR |
| **PatternDetector** | Chart pattern recognition | Head & Shoulders, VCP, Double Top/Bottom |
| **AlphaGenerator** | Signal aggregation | Multi-timeframe signal scoring |
| **KeyLevelDetector** | Support/Resistance detection | Swing levels, Fibonacci, Volume Profile |
| **SMCDetector** | Smart Money Concepts | Order Blocks, FVG, BOS/CHoCH, Liquidity |
| **RedisClient** | Cache persistence | Survives container restarts |

### Support/Resistance Detection (KeyLevelDetector)

Comprehensive key level detection system with 5 methods:

1. **Dynamic Swing High/Low Detection**
   - Identifies price points where price is highest/lowest within N candles before AND after
   - Configurable lookback period (default: 5 candles)
   - Works across all timeframes (1H, 4H, 1D)

2. **Fibonacci Retracement/Extension**
   - Retracement levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
   - Extension levels: 127.2%, 161.8%, 261.8%
   - Auto-calculated from detected swing points

3. **Level Strength Tracking**
   - Tracks how many times each level has been tested
   - Measures bounce rate (successful holds vs breaks)
   - Strength grades: Weak (1-2 tests), Moderate (3-4), Strong (5+), Very Strong (7+ with high bounce rate)

4. **Multi-Timeframe S/R Clustering**
   - Combines nearby levels from different timeframes
   - Clustered levels weighted by timeframe significance (1D > 4H > 1H)
   - Configurable cluster threshold (default: 0.5%)

5. **Volume Profile S/R**
   - High Volume Nodes (HVN): Areas of high trading activity → support/resistance
   - Low Volume Nodes (LVN): Areas of low activity → price moves quickly through
   - 50-bin volume histogram analysis

### Smart Money Concepts (SMCDetector)

Comprehensive Smart Money Concepts detection for institutional trading patterns:

| Concept | Description | Trading Use |
|---------|-------------|-------------|
| **Order Blocks (OB)** | Last opposing candle before impulse move | Institutional entry zones |
| **Fair Value Gaps (FVG)** | Price imbalances/inefficiencies (3-candle gaps) | Price magnets for retracement |
| **Break of Structure (BOS)** | Price breaking swing levels in trend direction | Trend continuation confirmation |
| **Change of Character (CHoCH)** | First break against prevailing trend | Early reversal signal |
| **Liquidity Pools** | Equal highs/lows, swing points (stop clusters) | Stop hunt / trap detection |
| **Premium/Discount Zones** | Value areas based on Fibonacci 50% | Entry timing (buy discount, sell premium) |
| **Inducement** | False breakouts / stop hunts | Trap detection and reversal confirmation |

**Order Block Detection:**
- Bullish OB: Last red candle before strong bullish impulse (>1.5%)
- Bearish OB: Last green candle before strong bearish impulse (>1.5%)
- Strength based on impulse size: Weak (<2%), Moderate (2-3%), Strong (3-5%), Very Strong (>5%)

**Fair Value Gap Detection:**
- Bullish FVG: Gap where Candle[0].high < Candle[2].low
- Bearish FVG: Gap where Candle[0].low > Candle[2].high
- Minimum gap size: 0.3% for signal validity
- Tracks fill percentage as price returns to gap

**Market Structure Analysis:**
- Detects Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
- Trend determination: BULLISH (HH+HL), BEARISH (LH+LL), RANGING
- BOS: Break in trend direction (continuation)
- CHoCH: First break against trend (reversal)

**Premium/Discount Zones:**
- Premium Zone: Above 61.8% Fib (expensive - sell zone)
- Discount Zone: Below 38.2% Fib (cheap - buy zone)
- Equilibrium: Around 50% (fair value)

**Caching Strategy:**
- Order Blocks: 24h TTL
- FVGs: 24h TTL
- Market Structure: 1h TTL (changes frequently)
- Liquidity Pools: 24h TTL
- Premium/Discount: 4h TTL

### AI Agents

| Agent | Role | Strategy Logic |
|-------|------|----------------|
| **Regime Detector** | Classifies market (volatility, trend, volume) | EMA slopes, ATR, volume analysis |
| **Mean Reversion** | Buy low, sell high (fade extremes) | RSI + Bollinger Bands on 4H timeframe |
| **Trend Following** | Buy high, sell higher (ride trends) | VCP pattern, EMA (8/20/50) alignment |
| **Turtle Trading** | Classic breakout system | 20/55-day channel breakouts on daily candles |
| **Portfolio Manager** | Dynamic confidence gatekeeper | Adjusts threshold based on volatility, drawdown |
| **Risk Manager** | Position sizing, leverage control | Max 20x, drawdown protection |
| **Execution Agent** | Order optimization, AI logging | Limit vs market orders |

### The Two Strategies Framework

Based on research from professional traders:

**Mean Reversion** (Buy Low, Sell High)
- Assumes extremes don't last
- High win rate, low risk/reward
- Position sizing is the last line of defense
- P&L: Many small wins, few large losses

**Trend Following** (Buy High, Sell Higher)
- Trends persist longer than expected
- Low win rate, high reward per winner
- Strict stop-losses protect capital
- P&L: Many small losses, few big wins

### Why Regime-Based Routing?

1. **Adaptive Strategy**: Right tool for right market conditions
2. **Reduced Drawdowns**: Don't fight the market structure
3. **Clear Decision Trail**: AI logs show regime → strategy → action
4. **Research-Backed**: Based on proven professional trading wisdom

## Self-Evolving RL System

Our system implements a **Reflexion-style** architecture that learns from every trade and autonomously evolves parameters within safe bounds.

### Triple Memory System (Redis)

| Memory Type | Purpose | TTL |
|-------------|---------|-----|
| **Episodic** | Complete trade records with context, outcome, reflection | 90 days |
| **Semantic** | Extracted patterns and insights from trade history | 1 year |
| **Procedural** | Strategy parameters with evolution history | Never expires |

### Reflection Agents

| Agent | Role | Trigger |
|-------|------|---------|
| **ReflectionAgent** | Reviews open positions, suggests HOLD/CLOSE/REDUCE/ADD | Hourly |
| **JudgeAgent** | Scores completed trades on multi-objective reward | On trade close |
| **LearnerAgent** | Extracts patterns, evolves parameters autonomously | Daily consolidation |

### Learning Loops

| Loop | Frequency | Purpose |
|------|-----------|---------|
| **Position Monitor** | Every 60 seconds | Detects when positions are closed, triggers Trade Outcome Loop |
| **Position Review** | Every 1 hour | Check if positions should be adjusted based on regime changes |
| **Trade Outcome** | On close | Record P&L, score trade, generate LLM reflection, extract lessons |
| **Consolidation** | Daily | Extract patterns, generate insights, apply parameter evolutions |

### Autonomous Parameter Evolution

The Learner Agent can autonomously adjust parameters within safe bounds:

```python
PARAMETER_BOUNDS = {
    "portfolio_manager": {
        "base_confidence_threshold": (0.4, 0.8),  # Dynamic confidence
        "max_portfolio_exposure": (0.3, 0.6),      # Position limits
    },
    "mean_reversion": {
        "rsi_oversold": (20, 35),                  # Entry thresholds
        "rsi_overbought": (65, 80),
    },
    "trend_following": {
        "atr_multiplier": (1.5, 3.0),             # Stop distances
    },
    "turtle_trading": {
        "stop_atr_mult": (1.5, 3.0),
        "risk_per_trade": (0.005, 0.02),          # Position sizing
    },
}
```

### Multi-Objective Trade Scoring

The Judge Agent scores each trade on:
- **Return** (35%): Risk-adjusted return (Sharpe-like)
- **Risk Management** (25%): Max adverse excursion penalty
- **Execution** (15%): Slippage and fill quality
- **Timing** (15%): Entry/exit timing quality
- **Discipline** (10%): Adherence to strategy rules

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/whymelabs/whyme-quant-strategy-weex-ai.git
cd whyme-quant-strategy-weex-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your WEEX API credentials
```

Required variables:

```env
# WEEX API Credentials
WEEX_API_KEY=your-api-key
WEEX_SECRET_KEY=your-secret-key
WEEX_PASSPHRASE=your-passphrase
WEEX_API_URL=https://api-contract.weex.com

# WhyMe Quant Main Server (optional, for integration)
MAIN_SERVER_URL=https://whyme-quant.swmengappdev.workers.dev
NODE_API_KEY=your-node-api-key

# AI Model Configuration
OPENAI_API_KEY=your-openai-key  # For LLM agents
```

### 4. Run

```bash
# Development
python -m strategy_engine.main

# Or with Docker
docker-compose up -d
```

## Project Structure

```
whyme-quant-strategy-weex-ai/
├── docs/                      # Documentation
│   └── STRATEGY_RESEARCH.md  # Research insights & strategy rationale
├── strategy_engine/           # Trading Engine
│   ├── main.py               # Entry point (edge-based or regime-based)
│   ├── core/                 # Core abstractions
│   │   ├── base.py          # Signal & SignalAction classes
│   │   └── orchestrator.py  # Regime-based agent orchestrator
│   ├── config/              # Configuration (Edge Collection System)
│   │   ├── symbols.py        # Multi-symbol trading configuration
│   │   └── edges.py          # Predefined edge definitions
│   ├── models/              # Data Models
│   │   ├── memory.py         # TradeRecord, StrategyInsight, ParameterState
│   │   └── edge.py           # Edge, EdgeSignal, PositionSize (Edge Collection)
│   ├── services/            # Shared Services Layer
│   │   ├── redis_client.py   # Redis cache persistence
│   │   ├── market_data_service.py # Multi-timeframe OHLCV with caching
│   │   ├── indicators_service.py  # Technical indicator calculations
│   │   ├── pattern_detector.py    # Chart pattern detection
│   │   ├── alpha_generator.py     # Signal aggregation & scoring
│   │   ├── key_level_detector.py  # Support/Resistance detection (Swing, Fib, Volume Profile)
│   │   ├── trade_memory.py        # Triple Memory System (Episodic/Semantic/Procedural)
│   │   ├── edge_registry.py       # Edge storage & statistics (Edge Collection)
│   │   ├── kelly_sizer.py         # Kelly-based position sizing (Edge Collection)
│   │   ├── edge_scanner.py        # Edge signal scanning (Edge Collection)
│   │   └── performance_tracker.py # Attribution & convergence (Edge Collection)
│   ├── agents/              # AI Agents (Multi-Agent System)
│   │   ├── base_agent.py     # Base agent class
│   │   ├── regime_detector.py # Market regime classification
│   │   ├── mean_reversion.py # Mean reversion strategy (RSI, BB)
│   │   ├── trend_following.py # Trend following strategy (VCP, EMA)
│   │   ├── turtle_trading.py # Classic Turtle breakout (20/55 day)
│   │   ├── portfolio_manager.py # Dynamic confidence gatekeeper
│   │   ├── risk_manager.py   # Position sizing & risk control
│   │   ├── execution.py      # Order execution optimization
│   │   ├── reflection_agent.py  # Position review (Self-Evolving RL)
│   │   ├── judge_agent.py       # Trade scoring (Self-Evolving RL)
│   │   └── learner_agent.py     # Pattern extraction (Self-Evolving RL)
│   └── loops/               # Learning Loops (Self-Evolving RL)
│       ├── position_review_loop.py  # Hourly position review
│       ├── trade_outcome_loop.py    # On-close trade analysis
│       └── consolidation_loop.py    # Daily learning consolidation
├── weex_client/              # WEEX API Client
│   ├── client.py            # REST API client with AI log upload
│   └── auth.py              # HMAC-SHA256 authentication
├── ai_logging/               # AI Log System (Hackathon requirement)
│   ├── logger.py            # AI decision logger
│   ├── uploader.py          # WEEX AI log uploader
│   └── models.py            # Log data models
├── shared/                   # Shared Utilities
│   ├── config.py            # Pydantic settings
│   ├── discord.py           # Discord webhook notifications (insights, evolutions)
│   └── llm.py               # LLM analysis client (trade reflection, pattern extraction)
├── docker-compose.yml        # Docker setup (strategy + redis)
├── Dockerfile               # Strategy engine container
├── requirements.txt
└── README.md
```

## Key Features

### Multi-Timeframe Analysis
- **1H**: Short-term trend detection, entry timing
- **4H**: Medium-term signals, Mean Reversion indicators
- **1D**: Long-term trend, Turtle Trading breakouts

### Redis Persistence
- Candle data survives container restarts
- Fast startup (loads from cache vs API)
- TTL-based expiry (1D: 7 days, 4H: 3 days, 1H: 1 day)
- Automatic housekeeping every 5 minutes

### Real-Time Notifications
- Discord webhooks for trade signals, regime changes, errors
- LLM-powered trade analysis and explanations

## AI Logging (Critical for Hackathon)

Every trading decision must be logged with:

```python
{
    "orderId": 123456789,           # WEEX order ID
    "stage": "Strategy Generation",  # Decision stage
    "model": "gpt-4-turbo",          # AI model used
    "input": {                       # Input to AI
        "prompt": "Analyze BTC/USDT...",
        "market_data": {...}
    },
    "output": {                      # AI output
        "action": "BUY",
        "confidence": 0.85,
        "reasoning": "..."
    },
    "explanation": "Based on market analysis..."  # Natural language
}
```

## Hackathon Requirements Checklist

- [ ] Connect to WEEX API (pass connection test)
- [ ] Execute minimum 10 trades
- [ ] Submit AI logs for all trades
- [ ] Stay within 20x leverage limit
- [ ] Public GitHub repository
- [ ] Documentation of trading logic
- [ ] AI participation description

## API Documentation

- [WEEX AI Wars API](https://www.weex.com/api-doc/ai/intro)
- [Participant Guide](https://www.weex.com/api-doc/ai/introduction/ParticipantGuide)

## Resources

- **DoraHacks**: https://dorahacks.io/hackathon/weex-forked-entry/detail
- **Telegram Group**: https://t.me/weexaiwars
- **Technical Support**: https://t.me/weexaiwars/1

## License

Internal use only - WhyMe Labs
