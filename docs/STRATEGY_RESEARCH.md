# Strategy Research - WEEX AI Hackathon

Research insights gathered for building our multi-agent AI trading strategy for the WEEX AI Hackathon.

## Executive Summary

After analyzing multiple trading strategies and multi-agent systems, the key insight is:

**All trading strategies fall into two buckets:**
1. **Mean Reversion** - Buy Low, Sell High
2. **Trend Following** - Buy High, Sell Higher

The winning approach: **Use BOTH strategies but apply them in different market regimes.**

---

## Key Strategy Insights

### 1. The Two Strategies Framework (@hackertrader - Niv Goren)

**Core Insight:** Despite countless indicators and setups, all profitable trading reduces to two fundamental approaches.

#### Mean Reversion (Buy Low, Sell High)
- **Assumption:** Extremes don't last
- **Entry:** Big drop → bet on bounce; big run-up → fade it
- **Win Rate:** Very high success rates
- **Risk/Reward:** Low risk/reward ratio
- **Position Duration:** Short holding periods
- **Stop-Loss:** Often just a guideline (further drops = better opportunity)
- **P&L Profile:** Many small wins, few large losses
- **Risk Management:** "Size is the last line of defense"
- **Exit Rules:** Rule-based (e.g., first green candle, 5% bounce)
- **Example:** Jason Shapiro (34%/year) - fades positioning + failed news using COT data

#### Trend Following (Buy High, Sell Higher)
- **Assumption:** Trends persist
- **Entry:** Buy what's already rising
- **Win Rate:** Wins less often
- **Risk/Reward:** Winners are much larger than losers
- **Position Duration:** Long holding periods
- **Stop-Loss:** Strict, protects against reversals
- **P&L Profile:** Many small losses, few big wins
- **Key Rule:** Take EVERY trade, avoid taking profits too early
- **Exit Rules:** Exit only when trend clearly reverses
- **Example:** Turtle Traders ($175M) - Buy when price closes above 20-day high

#### The Meta-Strategy
> "Some regimes reward trend following. Others reward mean reversion. Running both across different universes smooths returns, reduces drawdowns, and protects the account."

**For Our Strategy:** Build a regime-detection system that switches between Mean Reversion and Trend Following based on market conditions.

---

### 2. Swing Trading System (@felipeguirao - Felipe Guirao)

**Performance:** 100%+ per year using systematic swing trading

#### Technical Setup
- **EMAs:** 8, 20, 50 period
- **Pattern:** VCP (Volatility Contraction Pattern)
- **Consolidation:** Minimum 7 candles before breakout
- **Structure:** Higher-low mandatory before entry

#### Entry Rules
1. Wait for VCP pattern to form
2. Confirm 7+ candle consolidation
3. Identify higher-low structure
4. Enter on breakout at End of Day (EOD)

#### Key Principles
- Volatility contracts before expansion
- Don't chase - wait for proper setup
- Higher-low confirms buying pressure

**For Our Strategy:** Implement VCP detection with EMA confluence as a Trend Following signal generator.

---

### 3. Regime Detection with 36 Classifications (@web3tinkle)

**Concept:** Market exists in multiple regimes, each requiring a specific execution strategy.

#### 36 Regime Classifications
The system categorizes markets based on combinations of:
- **Volatility:** Low / Medium / High
- **Trend:** Strong Down / Weak Down / Sideways / Weak Up / Strong Up
- **Volume:** Low / Normal / High

Each combination (e.g., "High Volatility + Strong Up + High Volume") maps to a specific trading approach.

**For Our Strategy:** Build a regime classifier agent that categorizes current market state and routes to appropriate strategy (Mean Reversion vs Trend Following).

---

### 4. Multi-Agent Trading Architectures

#### MiroFish (@github_daily)
- Multi-agent simulation for stock prediction
- Each agent has independent "personality"
- Agents collaborate on final decision
- Simulates diverse trading perspectives

#### ai-trading-team (@discountifu)
- Open source modular architecture
- Multiple AI agents with specific roles
- Real-time coordination
- Extensible framework

**For Our Strategy:** Design agents with distinct personalities:
- **Conservative Risk Manager** - always prioritizes capital preservation
- **Aggressive Trend Follower** - looks for momentum opportunities
- **Contrarian Mean Reverter** - fades extreme moves
- **Orchestrator** - weighs inputs based on regime

---

### 5. Portfolio Optimization (@systematicls)

**Tools:** CVXPY + MOSEK for convex optimization

#### Key Concepts
- Optimize position sizing across multiple assets
- Balance risk-adjusted returns
- Consider correlation between positions
- Implement maximum drawdown constraints

**For Our Strategy:** Use optimization to:
1. Size positions based on confidence and volatility
2. Manage overall portfolio risk
3. Ensure diversification across strategies

---

### 6. Critical Implementation Insights

#### RSI/Bollinger Bands Need Higher Timeframes
**Problem:** RSI and Bollinger Bands generate too much noise on small timeframes (1m, 5m, 15m).

**Solution:** Multi-timeframe analysis
- Use higher timeframe (4H, Daily) for signal confirmation
- Lower timeframe for entry timing only
- Aggregate candles to build HTF data (e.g., 4 x 1H = 4H)

```
Lower Timeframe (1H) → Entry timing, quick reaction
Higher Timeframe (4H) → Signal confirmation, trend filter
```

#### Portfolio Management is Critical

**Problem:** Strategy signals are independent of portfolio context. A good signal in a bad portfolio context should be rejected.

**Solution:** Portfolio Manager as execution gatekeeper

The Portfolio Manager considers:
1. **Current positions** - exposure, correlation
2. **Market conditions** - volatility, regime
3. **Dynamic confidence threshold** - adjusts based on:
   - Portfolio exposure (higher exposure → higher threshold)
   - Recent win rate (losing streak → higher threshold)
   - Daily drawdown (approaching limit → higher threshold)
   - Volatility (high vol → higher threshold)

```
Signal → Portfolio Manager → Execute or Reject
         ↓
    Checks:
    - Confidence vs dynamic threshold
    - Portfolio exposure limits
    - Correlation with existing positions
    - Daily drawdown status
```

**Key Insight:** "A signal is just a suggestion. Portfolio context determines execution."

---

## Strategy Architecture for WEEX AI Hackathon

Based on the research, here's our multi-agent architecture with Portfolio Management:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     WEEX AI Strategy Engine                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐                           │
│  │  Market Data     │───▶│  Regime Detector │                           │
│  │  (Multi-TF)      │    │  Agent           │                           │
│  └──────────────────┘    └────────┬─────────┘                           │
│                                   │                                      │
│                    ┌──────────────┴───────────────┐                     │
│                    ▼                              ▼                      │
│  ┌──────────────────────┐      ┌──────────────────────┐                │
│  │  Mean Reversion      │      │  Trend Following     │                │
│  │  Strategy Agent      │      │  Strategy Agent      │                │
│  │  - HTF RSI/BB only   │      │  - Breakout entry    │                │
│  │  - Fade extremes     │      │  - Ride trends       │                │
│  │  (4H+ timeframe)     │      │  - VCP + EMA (8/20/50)│                │
│  └──────────┬───────────┘      └──────────┬───────────┘                │
│             │                              │                             │
│             └──────────────┬───────────────┘                            │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────┐               │
│  │           Portfolio Manager Agent                    │               │
│  │  (THE EXECUTION GATEKEEPER)                          │               │
│  │  - Dynamic confidence threshold                      │               │
│  │  - Portfolio exposure check                          │               │
│  │  - Correlation analysis                              │               │
│  │  - Drawdown protection                               │               │
│  │  - "Signal + Portfolio Context = Execute or Reject" │               │
│  └──────────────────────────┬──────────────────────────┘               │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────┐               │
│  │           Risk Manager Agent                         │               │
│  │  - Final position sizing                             │               │
│  │  - Leverage check (max 20x)                          │               │
│  └──────────────────────────┬──────────────────────────┘               │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────┐               │
│  │           Execution Agent                            │               │
│  │  - Order type optimization                           │               │
│  │  - AI Log recording (mandatory)                      │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

```
1. Market Data → Regime Detector
   "Is this a trending or ranging market?"

2. Regime → Strategy Selection
   Trending → Trend Following Agent
   Ranging  → Mean Reversion Agent (with HTF confirmation)
   Neutral  → Check both, take stronger signal

3. Strategy Signal → Portfolio Manager
   "Given my current positions and market conditions,
    should I execute this signal?"

   Dynamic threshold = base_threshold
                     + volatility_adjustment
                     + drawdown_adjustment
                     + win_rate_adjustment
                     + exposure_adjustment

   If signal_confidence < dynamic_threshold → REJECT
   If exceeds_exposure_limit → REJECT or REDUCE_SIZE
   If correlates_with_existing → REDUCE_SIZE

4. Portfolio-Approved Signal → Risk Manager
   Final position sizing within leverage limits

5. Sized Order → Execution Agent
   Order type selection, AI log recording
```

## Implementation Priorities

### Phase 1: Core Infrastructure
- [x] WEEX API client with authentication
- [x] AI logging system for hackathon compliance
- [x] Basic multi-agent orchestrator

### Phase 2: Strategy Agents
- [ ] Regime Detector Agent
  - Volatility calculation (ATR, Bollinger Width)
  - Trend identification (EMA slope, ADX)
  - Volume analysis

- [ ] Mean Reversion Agent
  - Extreme detection (Bollinger Bands, RSI)
  - Failed news/event detection
  - Position sizing for high win-rate strategy

- [ ] Trend Following Agent
  - Breakout detection (20-day high/low)
  - VCP pattern recognition
  - EMA crossover signals (8/20/50)

### Phase 3: Risk & Execution
- [ ] Portfolio optimizer with CVXPY
- [ ] Smart order routing
- [ ] Comprehensive AI decision logging

## Key Metrics for Hackathon

Per WEEX AI Hackathon requirements:
- [ ] API connection test passed
- [ ] AI logs uploaded (mandatory)
- [ ] Minimum 10 trades executed
- [ ] Maximum 20x leverage respected
- [ ] All trades with AI reasoning logged

---

## References

1. @hackertrader - "There Are Only 2 Trading Strategies in the World"
2. @felipeguirao - Swing Trading System (100%+/year)
3. @web3tinkle - 36 Regime Classification System
4. @github_daily - MiroFish Multi-Agent Simulation
5. @discountifu - ai-trading-team Open Source Framework
6. @systematicls - Portfolio Optimization with CVXPY
7. Turtle Traders - Original Trend Following System
8. Jason Shapiro (Market Wizards) - COT-based Mean Reversion

---

*Last Updated: January 2026*
*For: WEEX AI Hackathon (Deadline: Jan 18, 2026)*
