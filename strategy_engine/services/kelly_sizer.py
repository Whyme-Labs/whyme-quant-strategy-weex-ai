"""Kelly Sizer Service for Statistical Edge Collection System.

Implements Kelly Criterion-based position sizing:
- Calculate optimal position size based on edge statistics
- Apply fractional Kelly for safety
- Volatility adjustments
- Portfolio correlation limits
- BOOTSTRAP MODE: Warming edges trade with minimum size to collect data

Kelly Criterion formula:
    kelly = (win_rate * payoff_ratio - loss_rate) / payoff_ratio

Where:
    - win_rate: probability of winning
    - loss_rate: 1 - win_rate
    - payoff_ratio: avg_win / avg_loss

Fractional Kelly:
    position_pct = kelly * kelly_fraction (typically 0.25 for safety)
"""

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from ..models.edge import Edge, EdgeStatus, PositionSize


class KellySizer:
    """Position sizing based on Kelly Criterion.

    Calculates optimal position sizes using each edge's statistical properties,
    with safety limits and adjustments for volatility and correlation.

    Key principles:
    1. Position size proportional to edge magnitude
    2. Use fractional Kelly (25%) for safety against estimation errors
    3. Reduce size in high volatility
    4. Respect portfolio correlation limits
    5. BOOTSTRAP MODE: Warming edges trade with minimum size to collect data
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,  # Use 25% of full Kelly (conservative)
        max_position_pct: float = 0.10,  # Never more than 10% per trade
        min_position_pct: float = 0.01,  # Minimum 1% to be worth trading
        max_total_exposure: float = 0.50,  # Max 50% total portfolio exposure
        bootstrap_position_pct: float = 0.02,  # 2% for warming edges (bootstrap)
    ):
        """Initialize Kelly Sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_position_pct: Maximum position size as % of equity
            min_position_pct: Minimum position size to trade
            max_total_exposure: Maximum total portfolio exposure
            bootstrap_position_pct: Position size for warming edges (bootstrap mode)
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_total_exposure = max_total_exposure
        self.bootstrap_position_pct = bootstrap_position_pct

        logger.info(
            f"KellySizer initialized: fraction={kelly_fraction:.0%}, "
            f"max={max_position_pct:.0%}, min={min_position_pct:.0%}, "
            f"bootstrap={bootstrap_position_pct:.0%}"
        )

    def calculate_position_size(
        self,
        edge: Edge,
        account_equity: float,
        current_price: float,
        volatility_adjustment: float = 1.0,
        correlation_adjustment: float = 1.0,
        current_exposure: float = 0.0,
    ) -> PositionSize:
        """Calculate position size based on edge statistics.

        For WARMING edges: Uses bootstrap_position_pct (fixed small size)
        For ACTIVE edges: Uses Kelly Criterion for optimal sizing

        Args:
            edge: Edge object with statistical properties
            account_equity: Current account equity in USD
            current_price: Current price of the asset
            volatility_adjustment: Volatility adjustment factor (0-1)
            correlation_adjustment: Correlation adjustment factor (0-1)
            current_exposure: Current portfolio exposure as fraction

        Returns:
            PositionSize object with calculated size and reasoning
        """
        # Check basic validation
        if edge.status == EdgeStatus.DISABLED:
            return self._no_trade_position(edge, "Edge is disabled")

        if edge.status == EdgeStatus.PAUSED:
            return self._no_trade_position(edge, f"Edge is paused: {edge.pause_reason or 'unknown'}")

        # BOOTSTRAP MODE: Warming edges trade with minimum size
        if edge.status == EdgeStatus.WARMING:
            return self._bootstrap_position(
                edge=edge,
                account_equity=account_equity,
                current_price=current_price,
                volatility_adjustment=volatility_adjustment,
                current_exposure=current_exposure,
            )

        # ACTIVE edges: Full Kelly sizing
        return self._kelly_position(
            edge=edge,
            account_equity=account_equity,
            current_price=current_price,
            volatility_adjustment=volatility_adjustment,
            correlation_adjustment=correlation_adjustment,
            current_exposure=current_exposure,
        )

    def _bootstrap_position(
        self,
        edge: Edge,
        account_equity: float,
        current_price: float,
        volatility_adjustment: float,
        current_exposure: float,
    ) -> PositionSize:
        """Calculate bootstrap position for WARMING edges.

        Uses a fixed small position size to collect sample data.
        Once edge has enough samples, it will transition to ACTIVE.
        """
        # Use bootstrap size with volatility adjustment
        position_pct = self.bootstrap_position_pct * volatility_adjustment

        # Check portfolio exposure limit
        available_exposure = self.max_total_exposure - current_exposure
        if position_pct > available_exposure:
            if available_exposure <= 0:
                return self._no_trade_position(
                    edge, "Portfolio exposure limit reached"
                )
            position_pct = available_exposure

        # Convert to units
        position_value = account_equity * position_pct
        position_size = position_value / current_price if current_price > 0 else 0

        progress = f"{edge.sample_size}/{edge.min_sample_size}"
        reason = f"Bootstrap mode ({progress} samples): {position_pct:.1%} of equity"

        logger.info(
            f"Edge {edge.edge_id} BOOTSTRAP: {position_pct:.1%} "
            f"(samples: {progress})"
        )

        return PositionSize(
            size=position_size,
            size_usd=position_value,
            position_pct=position_pct,
            kelly_raw=0,  # No Kelly calc for bootstrap
            kelly_fractional=0,
            edge_expectancy=edge.expectancy,
            edge_win_rate=edge.win_rate,
            edge_payoff=edge.payoff_ratio,
            volatility_adjustment=volatility_adjustment,
            correlation_adjustment=1.0,
            capped_by_max=False,
            capped_by_correlation=current_exposure > 0,
            reason=reason,
            can_trade=True,
        )

    def _kelly_position(
        self,
        edge: Edge,
        account_equity: float,
        current_price: float,
        volatility_adjustment: float,
        correlation_adjustment: float,
        current_exposure: float,
    ) -> PositionSize:
        """Calculate Kelly-based position for ACTIVE edges."""

        # Validate edge has required data
        if edge.avg_loss_pct <= 0:
            return self._no_trade_position(edge, "Edge has no loss data")

        # Check if edge is profitable
        if edge.expectancy < edge.min_expectancy:
            return self._no_trade_position(
                edge,
                f"Edge below minimum expectancy: {edge.expectancy:.2%} < {edge.min_expectancy:.2%}",
            )

        # Calculate Kelly fraction
        kelly_raw = self._calculate_kelly(edge)
        if kelly_raw <= 0:
            return self._no_trade_position(
                edge,
                f"Kelly criterion negative or zero: {kelly_raw:.2%}",
            )

        # Apply fractional Kelly
        kelly_fractional = kelly_raw * self.kelly_fraction

        # Start with Kelly-based position
        position_pct = kelly_fractional

        # Apply volatility adjustment (reduce in high vol)
        position_pct *= volatility_adjustment

        # Apply correlation adjustment (reduce if correlated positions exist)
        position_pct *= correlation_adjustment

        # Track if we hit limits
        capped_by_max = False
        capped_by_correlation = False

        # Check portfolio exposure limit
        available_exposure = self.max_total_exposure - current_exposure
        if position_pct > available_exposure:
            position_pct = max(0, available_exposure)
            capped_by_correlation = True

        # Clamp to position limits
        if position_pct > self.max_position_pct:
            position_pct = self.max_position_pct
            capped_by_max = True

        # Check minimum viable size
        if position_pct < self.min_position_pct:
            return self._no_trade_position(
                edge,
                f"Position too small: {position_pct:.2%} < {self.min_position_pct:.2%}",
            )

        # Convert to units
        position_value = account_equity * position_pct
        position_size = position_value / current_price if current_price > 0 else 0

        # Build reason string
        reason_parts = [f"Kelly sizing: {position_pct:.1%} of equity"]
        if capped_by_max:
            reason_parts.append(f"(capped by max {self.max_position_pct:.0%})")
        if capped_by_correlation:
            reason_parts.append(f"(capped by exposure limit)")
        if volatility_adjustment < 1.0:
            reason_parts.append(f"(vol adj: {volatility_adjustment:.0%})")

        return PositionSize(
            size=position_size,
            size_usd=position_value,
            position_pct=position_pct,
            kelly_raw=kelly_raw,
            kelly_fractional=kelly_fractional,
            edge_expectancy=edge.expectancy,
            edge_win_rate=edge.win_rate,
            edge_payoff=edge.payoff_ratio,
            volatility_adjustment=volatility_adjustment,
            correlation_adjustment=correlation_adjustment,
            capped_by_max=capped_by_max,
            capped_by_correlation=capped_by_correlation,
            reason=" ".join(reason_parts),
            can_trade=True,
        )

    def _calculate_kelly(self, edge: Edge) -> float:
        """Calculate raw Kelly fraction.

        Kelly = (p * b - q) / b

        Where:
            p = win rate
            q = 1 - p (loss rate)
            b = payoff ratio (avg_win / avg_loss)

        Args:
            edge: Edge with statistical properties

        Returns:
            Raw Kelly fraction (can be > 1)
        """
        p = edge.win_rate
        q = 1 - p
        b = edge.payoff_ratio

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b

        return kelly

    def _no_trade_position(self, edge: Edge, reason: str) -> PositionSize:
        """Create a no-trade PositionSize.

        Args:
            edge: Edge object
            reason: Reason for not trading

        Returns:
            PositionSize with size=0
        """
        return PositionSize(
            size=0,
            size_usd=0,
            position_pct=0,
            kelly_raw=edge.kelly_fraction if edge else 0,
            kelly_fractional=0,
            edge_expectancy=edge.expectancy if edge else 0,
            edge_win_rate=edge.win_rate if edge else 0,
            edge_payoff=edge.payoff_ratio if edge else 0,
            reason=reason,
            can_trade=False,
        )

    # =========================================================================
    # VOLATILITY ADJUSTMENT
    # =========================================================================

    def get_volatility_adjustment(self, atr_pct: float) -> float:
        """Calculate volatility adjustment factor.

        Reduces position size in high volatility environments to maintain
        consistent risk exposure.

        Args:
            atr_pct: ATR as percentage of price

        Returns:
            Adjustment factor (0-1)
        """
        # ATR as % of price thresholds
        if atr_pct < 0.02:      # < 2% daily
            return 1.0          # Full size
        elif atr_pct < 0.04:    # 2-4% daily
            return 0.8          # 80% size
        elif atr_pct < 0.06:    # 4-6% daily
            return 0.6          # 60% size
        elif atr_pct < 0.08:    # 6-8% daily
            return 0.4          # 40% size
        else:                   # > 8% daily (extreme vol)
            return 0.25         # 25% size

    def get_correlation_adjustment(
        self,
        edge: Edge,
        existing_positions: Dict[str, float],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> float:
        """Calculate correlation adjustment factor.

        Reduces position size when correlated assets already have exposure.

        Args:
            edge: Edge for new position
            existing_positions: Dict of symbol -> exposure fraction
            correlation_matrix: Optional correlation matrix

        Returns:
            Adjustment factor (0-1)
        """
        if not existing_positions:
            return 1.0

        # Default correlation assumptions (crypto assets are highly correlated)
        default_correlations = {
            ("BTCUSDT", "ETHUSDT"): 0.85,
            ("BTCUSDT", "SOLUSDT"): 0.75,
            ("ETHUSDT", "SOLUSDT"): 0.80,
        }

        symbol = edge.symbol
        total_correlated_exposure = 0.0

        for pos_symbol, pos_exposure in existing_positions.items():
            if pos_symbol == symbol:
                # Same asset - full correlation
                corr = 1.0
            elif correlation_matrix and symbol in correlation_matrix:
                corr = correlation_matrix.get(symbol, {}).get(pos_symbol, 0.5)
            else:
                # Use default correlations
                key = tuple(sorted([symbol, pos_symbol]))
                corr = default_correlations.get(key, 0.5)

            total_correlated_exposure += pos_exposure * corr

        # Reduce new position based on correlated exposure
        # If already 30% correlated exposure, reduce new position by 30%
        adjustment = max(0.1, 1.0 - total_correlated_exposure)

        return adjustment

    # =========================================================================
    # PORTFOLIO ANALYSIS
    # =========================================================================

    def calculate_portfolio_size(
        self,
        edges: list,
        account_equity: float,
        prices: Dict[str, float],
        volatilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionSize]:
        """Calculate position sizes for multiple edges with correlation limits.

        Args:
            edges: List of Edge objects to size
            account_equity: Current account equity
            prices: Dict of symbol -> current price
            volatilities: Optional dict of symbol -> ATR as % of price

        Returns:
            Dict of edge_id -> PositionSize
        """
        results = {}
        current_exposure = 0.0
        existing_positions = {}

        # Sort edges: ACTIVE first, then by expectancy
        def sort_key(e):
            status_priority = 0 if e.status == EdgeStatus.ACTIVE else 1
            return (status_priority, -e.expectancy)

        sorted_edges = sorted(edges, key=sort_key)

        for edge in sorted_edges:
            symbol = edge.symbol
            price = prices.get(symbol, 0)
            vol = volatilities.get(symbol, 0.03) if volatilities else 0.03

            if price <= 0:
                results[edge.edge_id] = self._no_trade_position(
                    edge, f"No price for {symbol}"
                )
                continue

            vol_adj = self.get_volatility_adjustment(vol)
            corr_adj = self.get_correlation_adjustment(edge, existing_positions)

            position = self.calculate_position_size(
                edge=edge,
                account_equity=account_equity,
                current_price=price,
                volatility_adjustment=vol_adj,
                correlation_adjustment=corr_adj,
                current_exposure=current_exposure,
            )

            results[edge.edge_id] = position

            if position.can_trade:
                current_exposure += position.position_pct
                if symbol in existing_positions:
                    existing_positions[symbol] += position.position_pct
                else:
                    existing_positions[symbol] = position.position_pct

        return results

    def get_sizing_summary(
        self,
        positions: Dict[str, PositionSize],
    ) -> Dict[str, Any]:
        """Get summary of position sizing results.

        Args:
            positions: Dict of edge_id -> PositionSize

        Returns:
            Summary statistics
        """
        tradeable = [p for p in positions.values() if p.can_trade]
        non_tradeable = [p for p in positions.values() if not p.can_trade]

        total_exposure = sum(p.position_pct for p in tradeable)
        total_value = sum(p.size_usd for p in tradeable)

        avg_kelly = (
            sum(p.kelly_raw for p in tradeable) / len(tradeable)
            if tradeable else 0
        )

        capped_count = sum(
            1 for p in tradeable if p.capped_by_max or p.capped_by_correlation
        )

        bootstrap_count = sum(
            1 for p in tradeable if "Bootstrap" in p.reason
        )

        return {
            "total_edges": len(positions),
            "tradeable_edges": len(tradeable),
            "bootstrap_edges": bootstrap_count,
            "non_tradeable_edges": len(non_tradeable),
            "total_exposure_pct": total_exposure,
            "total_value_usd": total_value,
            "avg_kelly_raw": avg_kelly,
            "capped_positions": capped_count,
            "positions_by_reason": self._group_by_reason(non_tradeable),
        }

    def _group_by_reason(self, positions: list) -> Dict[str, int]:
        """Group non-tradeable positions by reason."""
        reasons = {}
        for p in positions:
            # Extract key reason
            reason_key = p.reason.split(":")[0] if ":" in p.reason else p.reason
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
        return reasons
