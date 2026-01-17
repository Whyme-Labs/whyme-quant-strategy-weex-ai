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
│                      WEEX AI Strategy Engine                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Services Layer (Multi-Timeframe + Redis)                │    │
│  │  MarketData → Indicators → PatternDetector → AlphaGenerator         │    │
│  │  (1H,4H,1D)    (EMA,RSI,BB)   (H&S,VCP)       (Aggregated Signals)  │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Agent Orchestrator                              │   │
│  │  Regime Detector → Strategy Selection → Portfolio Manager            │   │
│  │       │                    │                    │                     │   │
│  │       ▼                    ▼                    ▼                     │   │
│  │  (Trend/Range?)    ┌──────────────┐    (Dynamic Confidence)          │   │
│  │                    │ MeanRevert   │                                   │   │
│  │                    │ TrendFollow  │                                   │   │
│  │                    │ TurtleTrading│                                   │   │
│  │                    └──────────────┘                                   │   │
│  │                           │                                           │   │
│  │                    Risk Manager → Execution Agent → WEEX API          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Services Layer

1. **RedisClient** - Cache persistence for candle data (survives restarts)
2. **MarketDataService** - Multi-timeframe OHLCV (1H, 4H, 1D) with auto-refresh
3. **IndicatorsService** - Technical indicators (EMA, RSI, BB, ATR)
4. **PatternDetector** - Chart patterns (Head & Shoulders, VCP, Double Top/Bottom)
5. **AlphaGenerator** - Multi-timeframe signal aggregation

### Key Agents

1. **Regime Detector** - Classifies market into volatility/trend/volume regimes
2. **Mean Reversion Agent** - Fades extremes (RSI/BB on 4H timeframe)
3. **Trend Following Agent** - Rides trends (VCP, EMA alignment)
4. **Turtle Trading Agent** - Classic 20/55-day breakouts on daily candles
5. **Portfolio Manager** - Dynamic confidence threshold gatekeeper
6. **Risk Manager** - Position sizing and leverage checks
7. **Execution Agent** - Order optimization and AI log recording

## Key Directories

```
├── weex_client/           # WEEX API client
│   ├── client.py          # Main API client
│   └── auth.py            # Authentication (HMAC signing)
├── strategy_engine/       # Trading strategy
│   ├── main.py            # Entry point
│   ├── core/              # Orchestrator, base classes
│   ├── services/          # Shared services (NEW)
│   │   ├── redis_client.py      # Redis cache persistence
│   │   ├── market_data_service.py # Multi-timeframe OHLCV
│   │   ├── indicators_service.py  # Technical indicators
│   │   ├── pattern_detector.py    # Chart patterns
│   │   └── alpha_generator.py     # Signal aggregation
│   └── agents/            # AI agents
│       ├── turtle_trading.py      # Turtle breakout system
│       ├── portfolio_manager.py   # Dynamic gatekeeper
│       └── ...
├── ai_logging/            # AI decision logging (hackathon requirement)
├── shared/                # Shared utilities
│   ├── config.py          # Pydantic settings
│   ├── discord.py         # Discord notifications
│   └── llm.py             # LLM analysis
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
- **Redis Persistence**: Candle data survives container restarts (check `docker exec weex-redis redis-cli keys "candles:*"`)
- **Housekeeping**: Runs every 5 minutes, trims cache to limits
- Portfolio Manager acts as gatekeeper - rejects low-confidence signals
- Dynamic confidence threshold adjusts based on:
  - Market volatility
  - Daily drawdown
  - Recent win rate
  - Portfolio exposure

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
