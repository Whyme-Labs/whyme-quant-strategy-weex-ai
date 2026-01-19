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

### Docker (Recommended)

```bash
# Build and start all services
docker compose up -d --build

# View logs
docker compose logs -f strategy-engine

# Stop services
docker compose down

# Stop and remove volumes (clears Redis data)
docker compose down -v
```

**Services:**
- `weex-ai-strategy` - Strategy engine container
- `weex-redis` - Redis container for memory persistence

### Local Development (Code Changes Only)

**Important:** Local development CANNOT call the WEEX API due to IP whitelisting requirements. Only the production server IP is whitelisted.

```bash
# Create conda environment
conda create -n weex-ai python=3.11 -y

# Activate environment
conda activate weex-ai

# Install dependencies
pip install -r requirements.txt
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
├── Dockerfile             # Container build configuration
├── docker-compose.yml     # Multi-container orchestration
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
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Development Commands

### Docker Commands

```bash
# Build and start all services
docker compose up -d --build

# View strategy engine logs
docker compose logs -f strategy-engine

# View all logs
docker compose logs -f

# Restart strategy engine
docker compose restart strategy-engine

# Stop all services
docker compose down

# Rebuild after code changes
docker compose up -d --build

# Check container status
docker compose ps
```

### Local Development (Code Only - No API Access)

**Important:** WEEX API requires IP whitelisting. Only the production server IP (209.182.237.49) is whitelisted. Local development CANNOT make API calls.

```bash
# Activate environment
conda activate weex-ai

# Run unit tests (no API calls)
pytest tests/ -v
```

**Note:** Always deploy to the remote server to test with live WEEX API.

## Hackathon Requirements

- [ ] API connection verified
- [ ] AI logs uploaded for every trade (mandatory)
- [ ] Minimum 10 trades executed
- [ ] Maximum 20x leverage
- [ ] All trades with AI reasoning logged

## Important Notes

- **Production Only**: The strategy engine should ONLY run on the remote server (209.182.237.49), never locally
- **IP Whitelisting**: WEEX API requires IP whitelisting - only the production server IP is whitelisted
- **Docker Deployment**: Always use `docker compose` for running the strategy engine
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
- **WEEX UID:** `3004783944`

**Important:** The strategy engine should ONLY be run on the remote production server, never locally.

### Initial Setup (First Time)

```bash
# Connect to server
ssh root@209.182.237.49 -i ~/.ssh/ssdnodes_sg_1

# Clone repository
git clone https://github.com/Whyme-Labs/whyme-quant-strategy-weex-ai.git
cd whyme-quant-strategy-weex-ai

# Configure environment
cp .env.example .env
nano .env  # Edit with production credentials

# Build and start with Docker
docker compose up -d --build

# Verify containers are running
docker compose ps

# Check logs
docker compose logs -f strategy-engine
```

### Deployment (Code Updates)

```bash
# Connect to server
ssh root@209.182.237.49 -i ~/.ssh/ssdnodes_sg_1
cd whyme-quant-strategy-weex-ai

# Pull latest changes
git pull origin main

# Rebuild and restart
docker compose up -d --build

# Verify deployment
docker compose logs -f strategy-engine
```

### Management Commands

```bash
# View live logs
docker compose logs -f strategy-engine

# Restart strategy engine
docker compose restart strategy-engine

# Stop all services
docker compose down

# Stop and clear all data (including Redis)
docker compose down -v

# Check container status
docker compose ps

# Access Redis CLI
docker exec -it weex-redis redis-cli
```

## References

- [WEEX API Documentation](https://www.weex.com/api-doc/spot/introduction/APIBriefIntroduction)
- [WEEX API Domain](https://www.weex.com/api-doc/spot/QuickStart/APIDomain)
- [DoraHacks BUIDL](https://dorahacks.io/hackathon/weex-forked-entry)
