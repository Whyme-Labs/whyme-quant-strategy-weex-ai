# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

WEEX AI Strategy Engine - A multi-agent AI trading system for the WEEX AI Wars Hackathon. Uses regime-based strategy switching between Mean Reversion and Trend Following approaches.

**Hackathon:** WEEX AI Wars ($1.88M prize pool)
**Deadline:** January 18, 2026

## Tech Stack

- **Language:** Python 3.11
- **Async:** asyncio, httpx, aiohttp
- **Data:** pandas, numpy
- **AI/ML:** OpenAI, Anthropic, LangChain
- **Logging:** loguru
- **Testing:** pytest, pytest-asyncio

## Environment Setup

```bash
# Create conda environment
conda create -n weex-ai python=3.11 -y

# Activate environment
conda activate weex-ai

# Install dependencies (use env pip directly to avoid path issues)
/opt/homebrew/Caskroom/miniconda/base/envs/weex-ai/bin/pip install -r requirements.txt

# Or install core dependencies manually
/opt/homebrew/Caskroom/miniconda/base/envs/weex-ai/bin/pip install \
    python-dotenv loguru httpx pydantic pydantic-settings aiosqlite numpy pandas
```

**Conda Environment:** `weex-ai` (Python 3.11)

## Configuration

Copy `.env.example` to `.env` and fill in:
```
WEEX_API_KEY=your-api-key
WEEX_SECRET_KEY=your-secret-key
WEEX_PASSPHRASE=your-passphrase
WEEX_API_URL=https://api.weex.com
```

## Architecture

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

### Services Layer

1. **RedisClient** - Cache persistence for candle data and trade memory
2. **MarketDataService** - Multi-timeframe OHLCV (1H, 4H, 1D) with auto-refresh
3. **IndicatorsService** - Technical indicators (EMA, RSI, BB, ATR)
4. **PatternDetector** - Chart patterns (Head & Shoulders, VCP, Double Top/Bottom)
5. **AlphaGenerator** - Multi-timeframe signal aggregation
6. **TradeMemoryService** - Triple memory system (Episodic, Semantic, Procedural)

### Trading Agents

1. **Regime Detector** - Classifies market into volatility/trend/volume regimes
2. **Mean Reversion Agent** - Fades extremes (RSI/BB on 4H timeframe)
3. **Trend Following Agent** - Rides trends (VCP, EMA alignment)
4. **Turtle Trading Agent** - Classic 20/55-day breakouts on daily candles
5. **Portfolio Manager** - Dynamic confidence threshold gatekeeper
6. **Risk Manager** - Position sizing and leverage checks
7. **Execution Agent** - Order optimization and AI log recording

### Self-Evolving RL Agents

1. **Reflection Agent** - Reviews open positions hourly, suggests actions (HOLD/CLOSE/REDUCE/ADD)
2. **Judge Agent** - Scores completed trades on multi-objective reward (return, timing, risk management)
3. **Learner Agent** - Extracts patterns from trade history, evolves parameters within safe bounds

### Learning Loops

1. **Position Monitor** - Runs every 60 seconds, detects when positions are closed and triggers Trade Outcome Loop
2. **Position Review Loop** - Runs hourly, checks if positions should be adjusted based on regime changes
3. **Trade Outcome Loop** - Triggered on trade close, scores trade and generates LLM reflection
4. **Consolidation Loop** - Runs daily, extracts patterns and applies parameter evolutions

## Key Directories

```
├── weex_client/           # WEEX API client
│   ├── client.py          # Main API client
│   └── auth.py            # Authentication (HMAC signing)
├── strategy_engine/       # Trading strategy
│   ├── main.py            # Entry point
│   ├── core/              # Orchestrator, base classes
│   ├── models/            # Data models
│   │   └── memory.py      # TradeRecord, StrategyInsight, ParameterState
│   ├── services/          # Shared services
│   │   ├── redis_client.py      # Redis cache persistence
│   │   ├── market_data_service.py # Multi-timeframe OHLCV
│   │   ├── indicators_service.py  # Technical indicators
│   │   ├── pattern_detector.py    # Chart patterns
│   │   ├── alpha_generator.py     # Signal aggregation
│   │   └── trade_memory.py        # Triple memory system
│   ├── agents/            # AI agents
│   │   ├── turtle_trading.py      # Turtle breakout system
│   │   ├── portfolio_manager.py   # Dynamic gatekeeper
│   │   ├── reflection_agent.py    # Position review (Self-Evolving RL)
│   │   ├── judge_agent.py         # Trade scoring (Self-Evolving RL)
│   │   ├── learner_agent.py       # Pattern extraction (Self-Evolving RL)
│   │   └── ...
│   └── loops/             # Learning loops (Self-Evolving RL)
│       ├── position_review_loop.py   # Hourly position review
│       ├── trade_outcome_loop.py     # On-close trade analysis
│       └── consolidation_loop.py     # Daily learning consolidation
├── ai_logging/            # AI decision logging (hackathon requirement)
├── shared/                # Shared utilities
│   ├── config.py          # Pydantic settings
│   ├── discord.py         # Discord notifications
│   └── llm.py             # LLM analysis + reflection methods
├── scripts/               # Utility scripts
│   └── deploy.sh          # Production deployment
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Development Commands

```bash
# Activate environment
conda activate weex-ai

# Test connection
python scripts/test_connection.py

# Run tests
pytest tests/ -v

# Run strategy engine
python -m strategy_engine.main
```

## Hackathon Requirements

- [ ] API connection verified
- [ ] AI logs uploaded for every trade (mandatory)
- [ ] Minimum 10 trades executed
- [ ] Maximum 20x leverage
- [ ] All trades with AI reasoning logged

## Important Notes

- **Multi-Timeframe**: Use 1H for entry timing, 4H for medium signals, 1D for Turtle breakouts
- **Redis Persistence**: Candle data and trade memory survive container restarts
- **Housekeeping**: Runs every 5 minutes, trims cache to limits
- Portfolio Manager acts as gatekeeper - rejects low-confidence signals
- Dynamic confidence threshold adjusts based on:
  - Market volatility
  - Daily drawdown
  - Recent win rate
  - Portfolio exposure

## Self-Evolving RL System

### Triple Memory System (Redis)

| Memory Type | Purpose | TTL |
|-------------|---------|-----|
| **Episodic** | Trade records with full context | 90 days |
| **Semantic** | Extracted patterns and insights | 1 year |
| **Procedural** | Strategy parameters and evolution history | Never expires |

**Redis Key Schema:**
```
memory:episodic:{trade_id}           → TradeRecord JSON
memory:episodic:open                 → Set of open trade_ids
memory:semantic:{insight_id}         → StrategyInsight JSON
memory:procedural:{agent}:{param}    → ParameterState JSON
memory:procedural:evolution          → List of evolutions
```

### Parameter Evolution Bounds

The Learner Agent can autonomously adjust parameters within these bounds:

```python
PARAMETER_BOUNDS = {
    "portfolio_manager": {
        "base_confidence_threshold": (0.4, 0.8),
        "max_portfolio_exposure": (0.3, 0.6),
    },
    "mean_reversion": {
        "rsi_oversold": (20, 35),
        "rsi_overbought": (65, 80),
    },
    "trend_following": {
        "atr_multiplier": (1.5, 3.0),
    },
    "turtle_trading": {
        "stop_atr_mult": (1.5, 3.0),
        "risk_per_trade": (0.005, 0.02),
    },
}
```

### Verification Commands

```bash
# Check trade memory
docker exec weex-redis redis-cli keys "memory:episodic:*"

# Check insights
docker exec weex-redis redis-cli keys "memory:semantic:*"

# Check parameter states
docker exec weex-redis redis-cli keys "memory:procedural:*"

# Get open trades
docker exec weex-redis redis-cli smembers "memory:episodic:open"
```

## WEEX API Configuration

**Base URL:** `https://api-contract.weex.com`

**Endpoints (use `/capi/v2/` NOT `/api/v2/`):**
- Ticker: `GET /capi/v2/market/ticker?symbol=cmt_btcusdt`
- Orderbook: `GET /capi/v2/market/depth?symbol=cmt_btcusdt&type=step0`
- Klines: `GET /capi/v2/market/candles?symbol=cmt_btcusdt&granularity=1h`
- Account: `GET /capi/v2/account/account?symbol=cmt_btcusdt`
- Place Order: `POST /capi/v2/order/placeOrder`

**Symbol Format:**
- Use `cmt_btcusdt` NOT `BTCUSDT`
- The client auto-converts: `BTCUSDT` -> `cmt_btcusdt`

**Kline Granularity Values:**
`1m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `1w`, `1M`

**Note:** Use lowercase format (e.g., `1d` not `1D` or `1day`)

## Server Deployment

**Production Server:**
- **IP Address:** `209.182.237.49`
- **SSH Access:** `ssh root@209.182.237.49 -i ~/.ssh/ssdnodes_sg_1`

**Deployment Steps:**
```bash
# Connect to server
ssh root@209.182.237.49 -i ~/.ssh/ssdnodes_sg_1

# Clone repository
git clone https://github.com/Whyme-Labs/whyme-quant-strategy-weex-ai.git
cd whyme-quant-strategy-weex-ai

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with production credentials

# Run the strategy engine
python -m strategy_engine.main
```

**WEEX UID:** `3004783944`

## References

- [WEEX API Documentation](https://www.weex.com/api-doc/spot/introduction/APIBriefIntroduction)
- [WEEX API Domain](https://www.weex.com/api-doc/spot/QuickStart/APIDomain)
- [DoraHacks BUIDL](https://dorahacks.io/hackathon/weex-forked-entry)
