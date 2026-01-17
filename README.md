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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      WEEX AI Multi-Agent Strategy Engine                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Services Layer (Multi-Timeframe)                  │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐  │    │
│  │  │ MarketData   │ │ Indicators   │ │  Pattern   │ │    Alpha     │  │    │
│  │  │ Service      │ │ Service      │ │  Detector  │ │  Generator   │  │    │
│  │  │ (1H,4H,1D)   │ │ (EMA,RSI,BB) │ │ (H&S,VCP)  │ │ (Aggregated) │  │    │
│  │  └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └──────┬───────┘  │    │
│  │         └─────────────────┴───────────────┴──────────────┘          │    │
│  │                              │ Redis Persistence                    │    │
│  └──────────────────────────────┼──────────────────────────────────────┘    │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Agent Orchestrator                              │   │
│  │  ┌────────────────┐                                                   │   │
│  │  │ Regime Detector│ ──▶ Classifies: Trending / Ranging / Volatile    │   │
│  │  └───────┬────────┘                                                   │   │
│  │          │                                                            │   │
│  │   ┌──────┴──────┬─────────────┬────────────────┐                     │   │
│  │   ▼             ▼             ▼                ▼                      │   │
│  │ ┌────────┐ ┌─────────┐ ┌───────────┐ ┌─────────────────┐             │   │
│  │ │ Mean   │ │ Trend   │ │  Turtle   │ │ Portfolio Mgr   │             │   │
│  │ │Revert  │ │Following│ │ Trading   │ │ (Gatekeeper)    │             │   │
│  │ │(RSI/BB)│ │(VCP/EMA)│ │(20/55 Day)│ │ Dynamic Conf.   │             │   │
│  │ └────────┘ └─────────┘ └───────────┘ └─────────────────┘             │   │
│  │                           │                                           │   │
│  │   ┌───────────────────────┴───────────────────────┐                  │   │
│  │   ▼                                               ▼                   │   │
│  │ ┌────────────────┐                    ┌──────────────────┐           │   │
│  │ │ Risk Manager   │                    │ Execution Agent  │           │   │
│  │ │ (Max 20x Lev)  │                    │ (AI Log Upload)  │           │   │
│  │ └────────────────┘                    └──────────────────┘           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌─────────────────────────────┐                          │
│                    │        WEEX API Client       │                          │
│                    │  (REST + AI Log Upload)      │                          │
│                    └─────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   WEEX Exchange     │
                         │   (Futures API)     │
                         └─────────────────────┘
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
│   ├── services/            # Shared Services Layer
│   │   ├── redis_client.py   # Redis cache persistence
│   │   ├── market_data_service.py # Multi-timeframe OHLCV with caching
│   │   ├── indicators_service.py  # Technical indicator calculations
│   │   ├── pattern_detector.py    # Chart pattern detection
│   │   └── alpha_generator.py     # Signal aggregation & scoring
│   └── agents/              # AI Agents (Multi-Agent System)
│       ├── base_agent.py     # Base agent class
│       ├── regime_detector.py # Market regime classification
│       ├── mean_reversion.py # Mean reversion strategy (RSI, BB)
│       ├── trend_following.py # Trend following strategy (VCP, EMA)
│       ├── turtle_trading.py # Classic Turtle breakout (20/55 day)
│       ├── portfolio_manager.py # Dynamic confidence gatekeeper
│       ├── risk_manager.py   # Position sizing & risk control
│       ├── execution.py      # Order execution optimization
│       └── market_analyst.py # Legacy market analysis (fallback)
├── weex_client/              # WEEX API Client
│   ├── client.py            # REST API client with AI log upload
│   └── auth.py              # HMAC-SHA256 authentication
├── ai_logging/               # AI Log System (Hackathon requirement)
│   ├── logger.py            # AI decision logger
│   ├── uploader.py          # WEEX AI log uploader
│   └── models.py            # Log data models
├── shared/                   # Shared Utilities
│   ├── config.py            # Pydantic settings
│   ├── discord.py           # Discord webhook notifications
│   └── llm.py               # LLM analysis client
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
