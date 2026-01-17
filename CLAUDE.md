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
┌─────────────────────────────────────────────────────────────────────────┐
│                     WEEX AI Strategy Engine                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Market Data → Regime Detector → Strategy Selection → Portfolio Manager │
│                     │                    │                    │         │
│                     ▼                    ▼                    ▼         │
│              (Trend/Range?)    (Mean Reversion    (Dynamic confidence   │
│                                 or Trend Following) threshold)          │
│                                        │                    │           │
│                                        ▼                    ▼           │
│                              Risk Manager → Execution Agent → WEEX API  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Agents

1. **Regime Detector** - Classifies market into volatility/trend/volume regimes
2. **Mean Reversion Agent** - Fades extremes (RSI/BB on higher timeframes)
3. **Trend Following Agent** - Rides trends (breakouts, VCP patterns)
4. **Portfolio Manager** - Dynamic confidence threshold gatekeeper
5. **Risk Manager** - Position sizing and leverage checks
6. **Execution Agent** - Order optimization and AI log recording

## Key Directories

```
├── weex_client/           # WEEX API client
│   ├── client.py          # Main API client
│   └── auth.py            # Authentication (HMAC signing)
├── strategy_engine/       # Trading strategy
│   ├── core/              # Orchestrator, base classes
│   └── agents/            # AI agents
├── ai_logging/            # AI decision logging (hackathon requirement)
├── scripts/               # Utility scripts
│   └── test_connection.py # Connection test
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

- Use higher timeframes (4H+) for RSI/Bollinger Bands to reduce noise
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
`1min`, `5min`, `15min`, `30min`, `1h`, `4h`, `12h`, `1day`, `1week`

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
