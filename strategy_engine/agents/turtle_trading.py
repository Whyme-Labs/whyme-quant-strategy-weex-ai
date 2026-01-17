"""Turtle Trading Agent - Classic Breakout System.

Implements the original Turtle Trading rules from Richard Dennis:
- System 1: 20-day breakout (short-term)
- System 2: 55-day breakout (long-term)
- Position Sizing: 1 Unit = 1% risk / (N × Dollar per point)
- Pyramiding: Add up to 4 units at 0.5N intervals
- Stop Loss: 2N from entry (moves with pyramiding)
- Exit: 10-day low (System 1) or 20-day low (System 2)

Reference: "Way of the Turtle" by Curtis Faith
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
from loguru import logger

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService
    from ..services.indicators_service import IndicatorsService


class TurtleSystem(Enum):
    """Turtle Trading System type."""
    SYSTEM1 = "system1"  # 20-day breakout
    SYSTEM2 = "system2"  # 55-day breakout


@dataclass
class TurtlePosition:
    """Tracks a Turtle trading position with pyramiding."""
    symbol: str
    direction: str  # "long" or "short"
    system: TurtleSystem
    units: int  # Number of units (max 4)
    entries: List[Dict]  # List of {price, size, timestamp}
    stop_price: float
    n_value: float  # ATR (N) at entry


@dataclass
class TurtleSignal:
    """Signal from Turtle Trading strategy."""
    symbol: str
    direction: str  # "long", "short", "none"
    action: str  # "enter", "add", "exit", "none"
    system: TurtleSystem
    entry_price: float
    stop_price: float
    position_size_pct: float
    unit_number: int  # Which unit (1-4) for pyramiding
    n_value: float
    channel_high: float
    channel_low: float
    reasoning: str
    confidence: float


class TurtleTradingAgent(BaseAgent):
    """Turtle Trading Agent implementing classic breakout rules.

    Uses real daily candle data to calculate proper 20-day and 55-day
    breakout channels with ATR-based position sizing.
    """

    name = "turtle_trading"
    stage_name = "Turtle Trading"
    model_name = "turtle-classic"

    def __init__(
        self,
        config: Dict[str, Any],
        market_data_service: Optional["MarketDataService"] = None,
        indicators_service: Optional["IndicatorsService"] = None,
    ):
        """Initialize Turtle Trading Agent.

        Args:
            config: Agent configuration
            market_data_service: Service for fetching real candle data
            indicators_service: Service for indicator calculations
        """
        super().__init__(config)

        # Services
        self.market_data_service = market_data_service
        self.indicators_service = indicators_service

        # Turtle parameters
        self.system1_entry = config.get("system1_entry", 20)    # Days
        self.system1_exit = config.get("system1_exit", 10)      # Days
        self.system2_entry = config.get("system2_entry", 55)    # Days
        self.system2_exit = config.get("system2_exit", 20)      # Days
        self.atr_period = config.get("atr_period", 20)          # N calculation
        self.stop_atr_mult = config.get("stop_atr_mult", 2.0)   # 2N stop
        self.pyramid_interval = config.get("pyramid_interval", 0.5)  # 0.5N for adds
        self.max_units = config.get("max_units", 4)             # Max 4 units
        self.risk_per_trade = config.get("risk_per_trade", 0.01)  # 1% risk per unit
        self.prefer_system = config.get("prefer_system", "system2")  # Default system

        # Active positions (keyed by symbol)
        self.positions: Dict[str, TurtlePosition] = {}

        # Track if System 1 last trade was a winner (skip rule)
        self.system1_last_winner: Dict[str, bool] = {}

        # Price history for calculations (when no market data service)
        self._price_history: Dict[str, List[Dict]] = {}

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate Turtle signals.

        Args:
            context: Context with market_data and optional regime

        Returns:
            Signal result with direction, entry, stop, and position size
        """
        market_data = context.get("market_data", {})
        symbol = market_data.get("symbol", "BTCUSDT")
        current_price = market_data.get("price", 0)

        if not current_price:
            return {"signal": None, "reasoning": "No price data"}

        # Get daily candles for breakout calculations
        daily_df = await self._get_daily_candles(symbol)

        if daily_df is None or len(daily_df) < self.system2_entry + 5:
            # Store price in history for future use
            self._store_price(symbol, current_price, market_data.get("timestamp"))
            return {
                "signal": None,
                "reasoning": f"Insufficient daily data ({len(daily_df) if daily_df is not None else 0} candles, need {self.system2_entry + 5})",
            }

        # Calculate N (ATR)
        n_value = self._calculate_n(daily_df)

        # Calculate channels
        s1_high = daily_df["high"].iloc[-self.system1_entry-1:-1].max()
        s1_low = daily_df["low"].iloc[-self.system1_entry-1:-1].min()
        s2_high = daily_df["high"].iloc[-self.system2_entry-1:-1].max()
        s2_low = daily_df["low"].iloc[-self.system2_entry-1:-1].min()
        exit_high = daily_df["high"].iloc[-self.system1_exit-1:-1].max()
        exit_low = daily_df["low"].iloc[-self.system1_exit-1:-1].min()

        # Check for existing position
        position = self.positions.get(symbol)

        if position:
            # Check for exit or pyramid add
            signal = self._check_position_management(
                position=position,
                current_price=current_price,
                n_value=n_value,
                exit_high=exit_high,
                exit_low=exit_low,
            )
        else:
            # Check for new entry
            signal = self._check_entry(
                symbol=symbol,
                current_price=current_price,
                n_value=n_value,
                s1_high=s1_high,
                s1_low=s1_low,
                s2_high=s2_high,
                s2_low=s2_low,
            )

        if not signal:
            return {
                "signal": None,
                "reasoning": f"No signal. Price ${current_price:,.2f} within channels. "
                            f"S1: [{s1_low:,.2f}, {s1_high:,.2f}], "
                            f"S2: [{s2_low:,.2f}, {s2_high:,.2f}]",
                "confidence": 0.0,
            }

        # Build response
        return {
            "signal": {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": signal.entry_price,
                "stop_price": signal.stop_price,
                "position_size_pct": signal.position_size_pct,
                "strategy": f"turtle_{signal.system.value}",
                "initial_target": None,  # Turtles use trailing exit, not fixed target
            },
            "action": signal.action,
            "unit_number": signal.unit_number,
            "n_value": signal.n_value,
            "channel_high": signal.channel_high,
            "channel_low": signal.channel_low,
            "reasoning": signal.reasoning,
            "confidence": signal.confidence,
            "explanation": signal.reasoning,
        }

    async def _get_daily_candles(self, symbol: str):
        """Get daily candle data from market data service or fallback."""
        if self.market_data_service:
            try:
                df = await self.market_data_service.get_candles(symbol, "1d", limit=100)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Failed to get daily candles from service: {e}")

        # Fallback to stored history (less reliable)
        history = self._price_history.get(symbol, [])
        if len(history) >= self.system2_entry + 5:
            import pandas as pd
            df = pd.DataFrame(history)
            return df

        return None

    def _calculate_n(self, df) -> float:
        """Calculate N (ATR) for position sizing.

        The original Turtles used a 20-day exponential moving average of True Range.
        """
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # True Range
        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]

        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]),
            )

        # 20-day EMA of TR
        n = tr[-self.atr_period:].mean()
        return n

    def _check_entry(
        self,
        symbol: str,
        current_price: float,
        n_value: float,
        s1_high: float,
        s1_low: float,
        s2_high: float,
        s2_low: float,
    ) -> Optional[TurtleSignal]:
        """Check for new entry signals.

        System 1 (20-day): Skip rule applies if last trade was winner
        System 2 (55-day): Always trade on breakout
        """
        # Calculate position size (1 unit = 1% risk)
        # Unit = (Account * Risk%) / (N * $ per point)
        # Simplified: position_size_pct = risk_per_trade / (2 * n_value / current_price)
        dollar_risk = n_value * self.stop_atr_mult  # 2N stop distance in dollars
        position_size_pct = self.risk_per_trade

        # Check System 2 breakout (55-day) - always trade
        if current_price > s2_high:
            stop_price = current_price - (n_value * self.stop_atr_mult)

            # Create position
            self.positions[symbol] = TurtlePosition(
                symbol=symbol,
                direction="long",
                system=TurtleSystem.SYSTEM2,
                units=1,
                entries=[{"price": current_price, "size": position_size_pct}],
                stop_price=stop_price,
                n_value=n_value,
            )

            return TurtleSignal(
                symbol=symbol,
                direction="long",
                action="enter",
                system=TurtleSystem.SYSTEM2,
                entry_price=current_price,
                stop_price=stop_price,
                position_size_pct=position_size_pct,
                unit_number=1,
                n_value=n_value,
                channel_high=s2_high,
                channel_low=s2_low,
                reasoning=f"LONG S2: Price ${current_price:,.2f} broke 55-day high ${s2_high:,.2f}. "
                         f"Stop at ${stop_price:,.2f} (2N). N=${n_value:,.2f}",
                confidence=0.85,
            )

        if current_price < s2_low:
            stop_price = current_price + (n_value * self.stop_atr_mult)

            self.positions[symbol] = TurtlePosition(
                symbol=symbol,
                direction="short",
                system=TurtleSystem.SYSTEM2,
                units=1,
                entries=[{"price": current_price, "size": position_size_pct}],
                stop_price=stop_price,
                n_value=n_value,
            )

            return TurtleSignal(
                symbol=symbol,
                direction="short",
                action="enter",
                system=TurtleSystem.SYSTEM2,
                entry_price=current_price,
                stop_price=stop_price,
                position_size_pct=position_size_pct,
                unit_number=1,
                n_value=n_value,
                channel_high=s2_high,
                channel_low=s2_low,
                reasoning=f"SHORT S2: Price ${current_price:,.2f} broke 55-day low ${s2_low:,.2f}. "
                         f"Stop at ${stop_price:,.2f} (2N). N=${n_value:,.2f}",
                confidence=0.85,
            )

        # Check System 1 breakout (20-day) - skip rule applies
        skip_system1 = self.system1_last_winner.get(symbol, False)

        if not skip_system1:
            if current_price > s1_high:
                stop_price = current_price - (n_value * self.stop_atr_mult)

                self.positions[symbol] = TurtlePosition(
                    symbol=symbol,
                    direction="long",
                    system=TurtleSystem.SYSTEM1,
                    units=1,
                    entries=[{"price": current_price, "size": position_size_pct}],
                    stop_price=stop_price,
                    n_value=n_value,
                )

                return TurtleSignal(
                    symbol=symbol,
                    direction="long",
                    action="enter",
                    system=TurtleSystem.SYSTEM1,
                    entry_price=current_price,
                    stop_price=stop_price,
                    position_size_pct=position_size_pct,
                    unit_number=1,
                    n_value=n_value,
                    channel_high=s1_high,
                    channel_low=s1_low,
                    reasoning=f"LONG S1: Price ${current_price:,.2f} broke 20-day high ${s1_high:,.2f}. "
                             f"Stop at ${stop_price:,.2f} (2N). N=${n_value:,.2f}",
                    confidence=0.75,
                )

            if current_price < s1_low:
                stop_price = current_price + (n_value * self.stop_atr_mult)

                self.positions[symbol] = TurtlePosition(
                    symbol=symbol,
                    direction="short",
                    system=TurtleSystem.SYSTEM1,
                    units=1,
                    entries=[{"price": current_price, "size": position_size_pct}],
                    stop_price=stop_price,
                    n_value=n_value,
                )

                return TurtleSignal(
                    symbol=symbol,
                    direction="short",
                    action="enter",
                    system=TurtleSystem.SYSTEM1,
                    entry_price=current_price,
                    stop_price=stop_price,
                    position_size_pct=position_size_pct,
                    unit_number=1,
                    n_value=n_value,
                    channel_high=s1_high,
                    channel_low=s1_low,
                    reasoning=f"SHORT S1: Price ${current_price:,.2f} broke 20-day low ${s1_low:,.2f}. "
                             f"Stop at ${stop_price:,.2f} (2N). N=${n_value:,.2f}",
                    confidence=0.75,
                )

        return None

    def _check_position_management(
        self,
        position: TurtlePosition,
        current_price: float,
        n_value: float,
        exit_high: float,
        exit_low: float,
    ) -> Optional[TurtleSignal]:
        """Check for exit or pyramid add.

        Pyramiding: Add units at 0.5N intervals, up to 4 units
        Exit: 10-day exit for System 1, 20-day exit for System 2
        Stop: 2N from most recent entry (moves with pyramiding)
        """
        # Check stop loss
        if position.direction == "long" and current_price <= position.stop_price:
            self._close_position(position.symbol, is_winner=False)
            return TurtleSignal(
                symbol=position.symbol,
                direction="none",
                action="exit",
                system=position.system,
                entry_price=current_price,
                stop_price=0,
                position_size_pct=0,
                unit_number=0,
                n_value=n_value,
                channel_high=0,
                channel_low=0,
                reasoning=f"EXIT STOP: Long stopped out at ${current_price:,.2f}. "
                         f"Stop was ${position.stop_price:,.2f}",
                confidence=1.0,
            )

        if position.direction == "short" and current_price >= position.stop_price:
            self._close_position(position.symbol, is_winner=False)
            return TurtleSignal(
                symbol=position.symbol,
                direction="none",
                action="exit",
                system=position.system,
                entry_price=current_price,
                stop_price=0,
                position_size_pct=0,
                unit_number=0,
                n_value=n_value,
                channel_high=0,
                channel_low=0,
                reasoning=f"EXIT STOP: Short stopped out at ${current_price:,.2f}. "
                         f"Stop was ${position.stop_price:,.2f}",
                confidence=1.0,
            )

        # Check trailing exit (10-day or 20-day depending on system)
        if position.direction == "long":
            exit_trigger = exit_low if position.system == TurtleSystem.SYSTEM1 else exit_high
            if current_price < exit_trigger:
                avg_entry = self._get_avg_entry(position)
                is_winner = current_price > avg_entry
                self._close_position(position.symbol, is_winner=is_winner)
                return TurtleSignal(
                    symbol=position.symbol,
                    direction="none",
                    action="exit",
                    system=position.system,
                    entry_price=current_price,
                    stop_price=0,
                    position_size_pct=0,
                    unit_number=0,
                    n_value=n_value,
                    channel_high=0,
                    channel_low=exit_trigger,
                    reasoning=f"EXIT TRAILING: Price ${current_price:,.2f} below "
                             f"{'10' if position.system == TurtleSystem.SYSTEM1 else '20'}-day "
                             f"low ${exit_trigger:,.2f}. {'Winner' if is_winner else 'Loser'}",
                    confidence=1.0,
                )

        if position.direction == "short":
            exit_trigger = exit_high if position.system == TurtleSystem.SYSTEM1 else exit_low
            if current_price > exit_trigger:
                avg_entry = self._get_avg_entry(position)
                is_winner = current_price < avg_entry
                self._close_position(position.symbol, is_winner=is_winner)
                return TurtleSignal(
                    symbol=position.symbol,
                    direction="none",
                    action="exit",
                    system=position.system,
                    entry_price=current_price,
                    stop_price=0,
                    position_size_pct=0,
                    unit_number=0,
                    n_value=n_value,
                    channel_high=exit_trigger,
                    channel_low=0,
                    reasoning=f"EXIT TRAILING: Price ${current_price:,.2f} above "
                             f"{'10' if position.system == TurtleSystem.SYSTEM1 else '20'}-day "
                             f"high ${exit_trigger:,.2f}. {'Winner' if is_winner else 'Loser'}",
                    confidence=1.0,
                )

        # Check for pyramid add (0.5N interval, max 4 units)
        if position.units < self.max_units:
            last_entry = position.entries[-1]["price"]
            pyramid_distance = position.n_value * self.pyramid_interval

            if position.direction == "long":
                add_trigger = last_entry + pyramid_distance
                if current_price >= add_trigger:
                    return self._add_pyramid_unit(position, current_price, n_value)

            if position.direction == "short":
                add_trigger = last_entry - pyramid_distance
                if current_price <= add_trigger:
                    return self._add_pyramid_unit(position, current_price, n_value)

        return None

    def _add_pyramid_unit(
        self,
        position: TurtlePosition,
        current_price: float,
        n_value: float,
    ) -> TurtleSignal:
        """Add a pyramid unit to existing position.

        Position sizing: Same as initial unit
        Stop adjustment: Move stop to 2N from new entry
        """
        position_size_pct = self.risk_per_trade
        new_unit = position.units + 1

        # Update position
        position.entries.append({
            "price": current_price,
            "size": position_size_pct,
        })
        position.units = new_unit

        # Move stop to 2N from new entry
        if position.direction == "long":
            position.stop_price = current_price - (n_value * self.stop_atr_mult)
        else:
            position.stop_price = current_price + (n_value * self.stop_atr_mult)

        return TurtleSignal(
            symbol=position.symbol,
            direction=position.direction,
            action="add",
            system=position.system,
            entry_price=current_price,
            stop_price=position.stop_price,
            position_size_pct=position_size_pct,
            unit_number=new_unit,
            n_value=n_value,
            channel_high=0,
            channel_low=0,
            reasoning=f"PYRAMID ADD: Unit {new_unit}/{self.max_units} at ${current_price:,.2f}. "
                     f"Stop moved to ${position.stop_price:,.2f}. "
                     f"Avg entry: ${self._get_avg_entry(position):,.2f}",
            confidence=0.80,
        )

    def _get_avg_entry(self, position: TurtlePosition) -> float:
        """Get weighted average entry price."""
        total_size = sum(e["size"] for e in position.entries)
        if total_size == 0:
            return position.entries[0]["price"]
        weighted = sum(e["price"] * e["size"] for e in position.entries)
        return weighted / total_size

    def _close_position(self, symbol: str, is_winner: bool):
        """Close position and update skip rule."""
        position = self.positions.pop(symbol, None)
        if position and position.system == TurtleSystem.SYSTEM1:
            # Update skip rule for System 1
            self.system1_last_winner[symbol] = is_winner

    def _store_price(self, symbol: str, price: float, timestamp: Any):
        """Store price for fallback daily calculation."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append({
            "timestamp": timestamp,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0,
        })

        # Keep only last 100 entries
        if len(self._price_history[symbol]) > 100:
            self._price_history[symbol] = self._price_history[symbol][-100:]

    def get_input_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get summarized input for AI logging."""
        summary = super().get_input_summary(context)
        summary["strategy"] = "turtle_trading"
        summary["parameters"] = {
            "system1_entry": self.system1_entry,
            "system2_entry": self.system2_entry,
            "atr_period": self.atr_period,
            "stop_atr_mult": self.stop_atr_mult,
            "max_units": self.max_units,
        }
        if self.positions:
            summary["active_positions"] = {
                sym: {
                    "direction": pos.direction,
                    "units": pos.units,
                    "stop": pos.stop_price,
                }
                for sym, pos in self.positions.items()
            }
        return summary
