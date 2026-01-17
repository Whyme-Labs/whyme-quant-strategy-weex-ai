"""Edge Scanner Service for Statistical Edge Collection System.

Scans market for active edge signals across all symbols and timeframes.
Replaces the previous regime-based signal routing with direct edge condition checking.

Flow:
1. Get all active edges from registry
2. For each edge, check if entry conditions are met
3. Generate EdgeSignal for matching conditions
4. Return list of signals for position sizing and execution
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from .edge_registry import EdgeRegistry
from .market_data_service import MarketDataService
from .indicators_service import IndicatorsService
from ..models.edge import Edge, EdgeSignal, EdgeStatus, EdgeType
from ..config.symbols import get_symbol_config


class EdgeScanner:
    """Scans market for active edge signals.

    Checks all active edges against current market conditions and generates
    EdgeSignal objects for any matching conditions.
    """

    def __init__(
        self,
        edge_registry: EdgeRegistry,
        market_data: MarketDataService,
        indicators: IndicatorsService,
    ):
        """Initialize Edge Scanner.

        Args:
            edge_registry: EdgeRegistry service for getting active edges
            market_data: MarketDataService for price data
            indicators: IndicatorsService for technical indicators
        """
        self.edge_registry = edge_registry
        self.market_data = market_data
        self.indicators = indicators
        self._last_scan: Optional[datetime] = None
        self._scan_count = 0

    async def scan_all_edges(self) -> List[EdgeSignal]:
        """Scan all active edges across all symbols.

        Returns:
            List of EdgeSignal objects for edges with conditions met
        """
        self._scan_count += 1
        self._last_scan = datetime.now()
        signals = []

        try:
            # Get all active edges
            active_edges = await self.edge_registry.get_active_edges()

            if not active_edges:
                logger.debug("No active edges to scan")
                return []

            logger.debug(f"Scanning {len(active_edges)} active edges")

            # Group edges by symbol for efficient data fetching
            edges_by_symbol = self._group_by_symbol(active_edges)

            for symbol, edges in edges_by_symbol.items():
                # Get market data and indicators for this symbol
                market_data = await self._get_market_context(symbol)
                if not market_data:
                    logger.warning(f"No market data for {symbol}, skipping edges")
                    continue

                indicators_by_tf = await self._get_indicators(symbol, edges)

                for edge in edges:
                    try:
                        signal = await self._check_edge(edge, market_data, indicators_by_tf)
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        logger.error(f"Error checking edge {edge.edge_id}: {e}")

            if signals:
                logger.info(
                    f"Edge scan complete: {len(signals)} signal(s) from "
                    f"{len(active_edges)} edges"
                )

            return signals

        except Exception as e:
            logger.error(f"Edge scan failed: {e}")
            return []

    async def scan_symbol_edges(self, symbol: str) -> List[EdgeSignal]:
        """Scan edges for a specific symbol.

        Args:
            symbol: Symbol to scan (e.g., "BTCUSDT")

        Returns:
            List of EdgeSignal objects
        """
        active_edges = await self.edge_registry.get_active_edges(symbol=symbol)

        if not active_edges:
            return []

        market_data = await self._get_market_context(symbol)
        if not market_data:
            return []

        indicators_by_tf = await self._get_indicators(symbol, active_edges)

        signals = []
        for edge in active_edges:
            try:
                signal = await self._check_edge(edge, market_data, indicators_by_tf)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error checking edge {edge.edge_id}: {e}")

        return signals

    def _group_by_symbol(self, edges: List[Edge]) -> Dict[str, List[Edge]]:
        """Group edges by symbol.

        Args:
            edges: List of Edge objects

        Returns:
            Dict of symbol -> list of edges
        """
        grouped = {}
        for edge in edges:
            symbol = edge.symbol
            if symbol not in grouped:
                grouped[symbol] = []
            grouped[symbol].append(edge)
        return grouped

    async def _get_market_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market context for a symbol.

        Args:
            symbol: Symbol string

        Returns:
            Market data dictionary or None
        """
        try:
            # Get latest price data
            # This depends on MarketDataService implementation
            candles_1h = await self.market_data.get_candles(symbol, "1h", limit=1)
            candles_4h = await self.market_data.get_candles(symbol, "4h", limit=1)
            candles_1d = await self.market_data.get_candles(symbol, "1d", limit=1)

            if not candles_1h:
                return None

            latest = candles_1h[-1] if candles_1h else None

            return {
                "symbol": symbol,
                "price": latest.get("close", 0) if latest else 0,
                "open": latest.get("open", 0) if latest else 0,
                "high": latest.get("high", 0) if latest else 0,
                "low": latest.get("low", 0) if latest else 0,
                "volume": latest.get("volume", 0) if latest else 0,
                "timestamp": datetime.now(),
                "candles_1h": candles_1h,
                "candles_4h": candles_4h,
                "candles_1d": candles_1d,
            }

        except Exception as e:
            logger.error(f"Failed to get market context for {symbol}: {e}")
            return None

    async def _get_indicators(
        self,
        symbol: str,
        edges: List[Edge],
    ) -> Dict[str, Dict[str, Any]]:
        """Get indicators for all timeframes needed by edges.

        Args:
            symbol: Symbol string
            edges: List of edges to get indicators for

        Returns:
            Dict of timeframe -> indicators dict
        """
        # Collect all timeframes needed
        timeframes = set(edge.timeframe for edge in edges)

        indicators_by_tf = {}
        for tf in timeframes:
            try:
                indicators = await self.indicators.calculate_indicators(symbol, tf)
                indicators_by_tf[tf] = indicators or {}
            except Exception as e:
                logger.error(f"Failed to get indicators for {symbol} {tf}: {e}")
                indicators_by_tf[tf] = {}

        return indicators_by_tf

    async def _check_edge(
        self,
        edge: Edge,
        market_data: Dict[str, Any],
        indicators_by_tf: Dict[str, Dict[str, Any]],
    ) -> Optional[EdgeSignal]:
        """Check if edge conditions are met.

        Args:
            edge: Edge to check
            market_data: Current market data
            indicators_by_tf: Indicators by timeframe

        Returns:
            EdgeSignal if conditions met, None otherwise
        """
        # Get indicators for edge's timeframe
        indicators = indicators_by_tf.get(edge.timeframe, {})
        if not indicators:
            return None

        conditions = edge.entry_conditions
        current_price = market_data.get("price", 0)

        if current_price <= 0:
            return None

        # Check conditions based on edge type
        if edge.edge_type == EdgeType.MEAN_REVERSION:
            return await self._check_mean_reversion(edge, indicators, current_price, market_data)

        elif edge.edge_type == EdgeType.TREND_FOLLOWING:
            return await self._check_trend_following(edge, indicators, current_price, market_data)

        elif edge.edge_type == EdgeType.TURTLE:
            return await self._check_turtle(edge, indicators, current_price, market_data)

        elif edge.edge_type == EdgeType.BREAKOUT:
            return await self._check_breakout(edge, indicators, current_price, market_data)

        return None

    async def _check_mean_reversion(
        self,
        edge: Edge,
        indicators: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[EdgeSignal]:
        """Check mean reversion edge conditions.

        Args:
            edge: Edge to check
            indicators: Technical indicators
            current_price: Current price
            market_data: Market data

        Returns:
            EdgeSignal if conditions met
        """
        conditions = edge.entry_conditions
        rsi = indicators.get("rsi", 50)
        bb_lower = indicators.get("bb_lower", 0)
        bb_upper = indicators.get("bb_upper", float("inf"))
        bb_middle = indicators.get("bb_middle", current_price)
        atr = indicators.get("atr", 0)

        # Check RSI conditions
        if "rsi_below" in conditions:
            if rsi > conditions["rsi_below"]:
                return None

        if "rsi_above" in conditions:
            if rsi < conditions["rsi_above"]:
                return None

        if "rsi_extreme" in conditions:
            # RSI extreme means < 20 or > 80
            if not (rsi < 20 or rsi > 80):
                return None

        # Check Bollinger Band position
        if "bb_position" in conditions:
            bb_pos = conditions["bb_position"]
            if bb_pos == "below_lower" and current_price > bb_lower:
                return None
            elif bb_pos == "above_upper" and current_price < bb_upper:
                return None

        # Determine side
        side = conditions.get("side", "long")
        if side == "auto":
            side = "long" if rsi < 50 else "short"

        # Calculate entry and stop
        if side == "long":
            suggested_entry = current_price
            suggested_stop = current_price - (atr * 2) if atr > 0 else current_price * 0.98
        else:
            suggested_entry = current_price
            suggested_stop = current_price + (atr * 2) if atr > 0 else current_price * 1.02

        risk_pct = abs(current_price - suggested_stop) / current_price

        return EdgeSignal(
            signal_id=f"sig_{edge.edge_id}_{uuid.uuid4().hex[:8]}",
            edge_id=edge.edge_id,
            edge=edge,
            symbol=edge.symbol,
            side=side,
            price=current_price,
            market_data={
                "rsi": rsi,
                "bb_lower": bb_lower,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "atr": atr,
                **market_data,
            },
            suggested_entry=suggested_entry,
            suggested_stop=suggested_stop,
            risk_pct=risk_pct,
        )

    async def _check_trend_following(
        self,
        edge: Edge,
        indicators: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[EdgeSignal]:
        """Check trend following edge conditions.

        Args:
            edge: Edge to check
            indicators: Technical indicators
            current_price: Current price
            market_data: Market data

        Returns:
            EdgeSignal if conditions met
        """
        conditions = edge.entry_conditions
        atr = indicators.get("atr", 0)
        ema_8 = indicators.get("ema_8", current_price)
        ema_20 = indicators.get("ema_20", current_price)
        ema_50 = indicators.get("ema_50", current_price)
        high_20d = indicators.get("high_20d", current_price)
        low_20d = indicators.get("low_20d", current_price)

        side = None
        triggered = False

        # Check breakout conditions
        if "breakout" in conditions:
            breakout_period = conditions["breakout"]
            if breakout_period == "20d":
                if current_price > high_20d:
                    side = "long"
                    triggered = True
                elif current_price < low_20d:
                    side = "short"
                    triggered = True

        # Check EMA alignment
        if "ema_alignment" in conditions:
            alignment = conditions["ema_alignment"]
            if alignment == "8_20_50":
                # Bullish: 8 > 20 > 50
                if ema_8 > ema_20 > ema_50:
                    side = "long"
                    triggered = True
                # Bearish: 8 < 20 < 50
                elif ema_8 < ema_20 < ema_50:
                    side = "short"
                    triggered = True

        # Check VCP (simplified - ATR contraction)
        if "pattern" in conditions and conditions["pattern"] == "vcp":
            atr_ratio = indicators.get("atr_ratio", 1.0)
            if atr_ratio < conditions.get("atr_contraction", 0.6):
                side = "long"  # VCP is typically long-biased
                triggered = True

        if not triggered:
            return None

        # Override side if specified
        if conditions.get("side") != "auto":
            side = conditions.get("side", side)

        # Calculate entry and stop
        atr_mult = 2.0
        if side == "long":
            suggested_entry = current_price
            suggested_stop = current_price - (atr * atr_mult) if atr > 0 else current_price * 0.96
        else:
            suggested_entry = current_price
            suggested_stop = current_price + (atr * atr_mult) if atr > 0 else current_price * 1.04

        risk_pct = abs(current_price - suggested_stop) / current_price

        return EdgeSignal(
            signal_id=f"sig_{edge.edge_id}_{uuid.uuid4().hex[:8]}",
            edge_id=edge.edge_id,
            edge=edge,
            symbol=edge.symbol,
            side=side,
            price=current_price,
            market_data={
                "ema_8": ema_8,
                "ema_20": ema_20,
                "ema_50": ema_50,
                "high_20d": high_20d,
                "low_20d": low_20d,
                "atr": atr,
                **market_data,
            },
            suggested_entry=suggested_entry,
            suggested_stop=suggested_stop,
            risk_pct=risk_pct,
        )

    async def _check_turtle(
        self,
        edge: Edge,
        indicators: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[EdgeSignal]:
        """Check Turtle trading edge conditions.

        Args:
            edge: Edge to check
            indicators: Technical indicators
            current_price: Current price
            market_data: Market data

        Returns:
            EdgeSignal if conditions met
        """
        conditions = edge.entry_conditions
        atr = indicators.get("atr", 0)

        # Get the relevant highs/lows
        system = conditions.get("system", "1")
        breakout_period = conditions.get("breakout", "20d")

        if breakout_period == "20d":
            high = indicators.get("high_20d", current_price)
            low = indicators.get("low_20d", current_price)
        elif breakout_period == "55d":
            high = indicators.get("high_55d", current_price)
            low = indicators.get("low_55d", current_price)
        else:
            return None

        side = None
        triggered = False

        # Check breakout
        if current_price > high:
            side = "long"
            triggered = True
        elif current_price < low:
            side = "short"
            triggered = True

        if not triggered:
            return None

        # System 1: Skip if last S1 trade was winner (not implemented yet)
        # This would require checking trade history

        # Calculate entry and stop (Turtle uses 2N stop)
        atr_mult = 2.0
        if side == "long":
            suggested_entry = current_price
            suggested_stop = current_price - (atr * atr_mult) if atr > 0 else current_price * 0.96
        else:
            suggested_entry = current_price
            suggested_stop = current_price + (atr * atr_mult) if atr > 0 else current_price * 1.04

        risk_pct = abs(current_price - suggested_stop) / current_price

        return EdgeSignal(
            signal_id=f"sig_{edge.edge_id}_{uuid.uuid4().hex[:8]}",
            edge_id=edge.edge_id,
            edge=edge,
            symbol=edge.symbol,
            side=side,
            price=current_price,
            market_data={
                "system": system,
                f"high_{breakout_period}": high,
                f"low_{breakout_period}": low,
                "atr": atr,
                **market_data,
            },
            suggested_entry=suggested_entry,
            suggested_stop=suggested_stop,
            risk_pct=risk_pct,
        )

    async def _check_breakout(
        self,
        edge: Edge,
        indicators: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any],
    ) -> Optional[EdgeSignal]:
        """Check generic breakout edge conditions.

        Args:
            edge: Edge to check
            indicators: Technical indicators
            current_price: Current price
            market_data: Market data

        Returns:
            EdgeSignal if conditions met
        """
        # Similar to trend following breakout
        return await self._check_trend_following(edge, indicators, current_price, market_data)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_scan_stats(self) -> Dict[str, Any]:
        """Get scanner statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "scan_count": self._scan_count,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
        }
