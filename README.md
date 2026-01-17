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

## Strategy: Regime-Based Multi-Agent Trading

This strategy uses specialized AI agents that collaborate based on detected market regimes:

### Services Layer

| Service | Purpose | Features |
|---------|---------|----------|
| **MarketDataService** | Multi-timeframe OHLCV data | 1H, 4H, 1D candles with Redis persistence |
| **IndicatorsService** | Technical indicator calculations | EMA, RSI, Bollinger Bands, ATR |
| **PatternDetector** | Chart pattern recognition | Head & Shoulders, VCP, Double Top/Bottom |
| **AlphaGenerator** | Signal aggregation | Multi-timeframe signal scoring |
| **RedisClient** | Cache persistence | Survives container restarts |

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
│   ├── main.py               # Entry point
│   ├── core/                 # Core abstractions
│   │   ├── base.py          # Signal & SignalAction classes
│   │   └── orchestrator.py  # Regime-based agent orchestrator
│   ├── models/              # Data Models (Self-Evolving RL)
│   │   └── memory.py         # TradeRecord, StrategyInsight, ParameterState
│   ├── services/            # Shared Services Layer
│   │   ├── redis_client.py   # Redis cache persistence
│   │   ├── market_data_service.py # Multi-timeframe OHLCV with caching
│   │   ├── indicators_service.py  # Technical indicator calculations
│   │   ├── pattern_detector.py    # Chart pattern detection
│   │   ├── alpha_generator.py     # Signal aggregation & scoring
│   │   └── trade_memory.py        # Triple Memory System (Episodic/Semantic/Procedural)
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
