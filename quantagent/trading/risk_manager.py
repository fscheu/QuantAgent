"""
Risk Manager: Validates trades BEFORE execution.

Performs 5-point validation:
1. Capital available: cash >= trade_value
2. Position limit: trade_value <= 10% of portfolio_value
3. Daily loss: current_daily_pnl >= -5% of portfolio_value
4. Circuit breaker: not already triggered
5. Position management: prevent adding to existing positions
   (only allow closing or reversing positions)

All validation happens BEFORE broker execution.
If validation fails, order is rejected and never reaches broker.
"""

from datetime import datetime, date
from typing import Dict, Optional, Tuple
from sqlalchemy.orm import Session

from quantagent.models import Trade, OrderSide


class RiskManager:
    """Validates trades before execution and tracks daily P&L."""

    def __init__(
        self,
        portfolio_manager,  # PortfolioManager instance
        max_daily_loss_pct: float = 0.05,  # 5% daily loss limit
        max_position_pct: float = 0.10,  # 10% max position size
        db: Optional[Session] = None,
    ):
        """
        Initialize Risk Manager.

        Args:
            portfolio_manager: PortfolioManager instance (for capital/position checks)
            max_daily_loss_pct: Maximum daily loss percentage (default 5%)
            max_position_pct: Maximum position size percentage (default 10%)
            db: SQLAlchemy session for querying trades
        """
        self.portfolio = portfolio_manager
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.db = db
        self.circuit_breaker_triggered = False
        self.daily_pnl_tracker: Dict[date, float] = {}  # Reset daily

    def validate_trade(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate trade BEFORE execution.

        Called by OrderManager BEFORE broker.place_order()

        Args:
            symbol: Trading symbol
            side: Order side (BUY or SELL)
            qty: Quantity to buy/sell
            price: Current market price

        Returns:
            Tuple of (is_valid, rejection_reason)
            - (True, None) if trade is valid
            - (False, reason_string) if trade is invalid
        """
        trade_value = qty * price

        # Check 1: Capital available
        if self.portfolio.cash < trade_value:
            return (
                False,
                f"Insufficient capital: need ${trade_value:.2f}, have ${self.portfolio.cash:.2f}",
            )

        # Check 2: Position size <= 10% of portfolio
        portfolio_value = self.portfolio.get_total_value()
        max_position_value = portfolio_value * self.max_position_pct
        if trade_value > max_position_value:
            return (
                False,
                f"Position too large: ${trade_value:.2f} > max ${max_position_value:.2f} (10% limit)",
            )

        # Check 3: Daily loss limit not exceeded
        daily_pnl = self.get_daily_pnl()
        max_daily_loss = -(portfolio_value * self.max_daily_loss_pct)
        if daily_pnl < max_daily_loss:
            return (
                False,
                f"Daily loss limit exceeded: ${daily_pnl:.2f} < max loss ${max_daily_loss:.2f} (5% limit)",
            )

        # Check 4: Circuit breaker not triggered
        if self.circuit_breaker_triggered:
            return (False, "Circuit breaker is active - no more trades allowed today")

        # Check 5: Position management - prevent adding to existing positions
        # (only allow closing or reversing)
        if symbol in self.portfolio.positions:
            existing_pos = self.portfolio.positions[symbol]
            existing_qty = existing_pos["qty"]

            if existing_qty != 0:
                is_long_position = existing_qty > 0
                is_short_position = existing_qty < 0

                # Prevent adding to LONG position
                if is_long_position and side == OrderSide.BUY:
                    return (
                        False,
                        f"Position already open: LONG {abs(existing_qty):.6f} shares. "
                        f"Cannot add to existing LONG position (prevents over-concentration)",
                    )

                # Prevent adding to SHORT position
                if is_short_position and side == OrderSide.SELL:
                    return (
                        False,
                        f"Position already open: SHORT {abs(existing_qty):.6f} shares. "
                        f"Cannot add to existing SHORT position (prevents over-concentration)",
                    )

                # Allow closing/reversing positions (LONG→SELL, SHORT→BUY)

        return (True, None)

    def get_daily_pnl(self) -> float:
        """
        Get today's realized and unrealized P&L.

        Returns:
            Total P&L for today (sum of realized trades + unrealized positions)
        """
        today = date.today()

        # If no DB, use in-memory tracker
        if not self.db:
            return self.daily_pnl_tracker.get(today, 0.0)

        # Query realized trades from today
        from quantagent.models import Trade

        trades_today = (
            self.db.query(Trade)
            .filter(Trade.closed_at >= datetime.combine(today, datetime.min.time()))
            .all()
        )

        realized_pnl = sum(float(t.pnl) if t.pnl else 0.0 for t in trades_today)

        # Add unrealized P&L from open positions
        unrealized_pnl = self.portfolio.get_unrealized_pnl()

        return realized_pnl + unrealized_pnl

    def on_trade_executed(self, trade) -> None:
        """
        Called after trade is executed to update daily P&L tracking.

        Args:
            trade: Executed Trade object
        """
        today = date.today()

        # Update daily P&L tracker
        current_daily_pnl = self.daily_pnl_tracker.get(today, 0.0)
        trade_pnl = float(trade.pnl) if trade.pnl else 0.0
        self.daily_pnl_tracker[today] = current_daily_pnl + trade_pnl

        # Check if daily loss limit is now exceeded
        portfolio_value = self.portfolio.get_total_value()
        max_daily_loss = -(portfolio_value * self.max_daily_loss_pct)

        if self.daily_pnl_tracker[today] < max_daily_loss:
            self.circuit_breaker_triggered = True

    def reset_daily_tracker(self) -> None:
        """Reset daily P&L tracker (call at start of each day)."""
        self.circuit_breaker_triggered = False
        today = date.today()
        self.daily_pnl_tracker[today] = 0.0

    def check_circuit_breaker(self) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit breaker is active.

        Returns:
            Tuple of (is_active, reason_if_active)
        """
        if self.circuit_breaker_triggered:
            return (
                True,
                "Circuit breaker is active - daily loss limit exceeded",
            )
        return (False, None)
