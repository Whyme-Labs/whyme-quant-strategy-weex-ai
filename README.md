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
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEEX AI Multi-Agent Strategy                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐                           │
│  │  Market Data     │───▶│  Regime Detector │                           │
│  │  Collector       │    │  Agent           │                           │
│  └──────────────────┘    └────────┬─────────┘                           │
│                                   │                                      │
│                    ┌──────────────┴───────────────┐                     │
│                    ▼                              ▼                      │
│  ┌──────────────────────┐      ┌──────────────────────┐                │
│  │  Mean Reversion      │      │  Trend Following     │                │
│  │  Strategy Agent      │      │  Strategy Agent      │                │
│  │  - Fade extremes     │      │  - Breakout entry    │                │
│  │  - High win rate     │      │  - Ride trends       │                │
│  │  - RSI + Bollinger   │      │  - VCP + EMA (8/20/50)│                │
│  └──────────┬───────────┘      └──────────┬───────────┘                │
│             │                              │                             │
│             └──────────────┬───────────────┘                            │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────┐                   │
│  │           Risk Manager Agent                     │                   │
│  │  - Position sizing (max 20x leverage)            │                   │
│  │  - Drawdown protection                           │                   │
│  └──────────────────────────┬──────────────────────┘                   │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────┐                   │
│  │           Execution Agent                        │                   │
│  │  - Order type optimization                       │                   │
│  │  - AI Log recording (mandatory)                  │                   │
│  └─────────────────────────────────────────────────┘                   │
│                             │                                            │
│  ┌──────────────────────────┴──────────────────────┐                   │
│  │              AI Logging System                   │                   │
│  │  (Mandatory for hackathon verification)          │                   │
│  └─────────────────────────────────────────────────┘                   │
│                             │                                            │
│  ┌──────────────────────────┴──────────────────────┐                   │
│  │              WEEX API Client                     │                   │
│  └─────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  WEEX Exchange  │
                    │  (Futures API)  │
                    └─────────────────┘
```

## Strategy: Regime-Based Multi-Agent Trading

This strategy uses specialized AI agents that collaborate based on detected market regimes:

| Agent | Role | Strategy Logic |
|-------|------|----------------|
| **Regime Detector** | Classifies market (volatility, trend, volume) | EMA slopes, ATR, volume analysis |
| **Mean Reversion** | Buy low, sell high (fade extremes) | RSI + Bollinger Bands, high win rate |
| **Trend Following** | Buy high, sell higher (ride trends) | 20-day breakout, VCP pattern, EMA alignment |
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
│   └── agents/              # AI Agents (Multi-Agent System)
│       ├── base_agent.py     # Base agent class
│       ├── regime_detector.py # Market regime classification
│       ├── mean_reversion.py # Mean reversion strategy (RSI, BB)
│       ├── trend_following.py # Trend following strategy (VCP, EMA)
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
│   └── config.py            # Pydantic settings
├── docker-compose.yml        # Docker setup (strategy + redis)
├── Dockerfile               # Strategy engine container
├── requirements.txt
└── README.md
```

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
