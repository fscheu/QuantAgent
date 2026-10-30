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

import logging
from datetime import date, datetime
from typing import Callable, Dict, Optional, Tuple, Union

from sqlalchemy.orm import Session

from quantagent.models import OrderSide, Trade

logger = logging.getLogger(__name__)
_DEFAULT_HEAT = object()


class RiskManager:
    """Validates trades before execution and tracks daily P&L."""

    def __init__(
        self,
        portfolio_manager,  # PortfolioManager instance
        max_daily_loss_pct: float = 0.05,  # 5% daily loss limit
        max_position_pct: float = 0.10,  # 10% max position size
        db: Optional[Session] = None,
        clock: Optional[Callable[[], date]] = None,
        timeframe: Optional[str] = None,
        max_heat_pct: Optional[float] = _DEFAULT_HEAT,
    ):
        """
        Initialize Risk Manager.

        Args:
            portfolio_manager: PortfolioManager instance (for capital/position checks)
            max_daily_loss_pct: Maximum daily loss percentage (default 5%)
            max_position_pct: Maximum position size percentage (default 10%)
            db: SQLAlchemy session for querying trades
            clock: Callable returning date; defaults to date.today (wall-clock)
            timeframe: Timeframe string (e.g. '1d', '1h', '4h')
        """
        self.portfolio = portfolio_manager
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        from quantagent import settings
        self.max_heat_pct = settings.TRADING_MAX_HEAT_PCT if max_heat_pct is _DEFAULT_HEAT else max_heat_pct
        self.db = db
        self.clock = clock or date.today
        self.timeframe = timeframe
        self.circuit_breaker_triggered = False
        self.daily_pnl_tracker: Dict[date, float] = {}  # Reset daily
        self._current_date: Optional[date] = None
        self._position_epsilon = 1e-8

    def set_date(self, current_date: Union[date, datetime, None]) -> None:
        """Set current simulation date for daily P&L tracking."""
        if isinstance(current_date, datetime):
            self._current_date = current_date.date()
        else:
            self._current_date = current_date

    def _get_date(self) -> date:
        """Get effective date: set date, or clock, or wall-clock today."""
        if self._current_date is not None:
            return self._current_date
        d = self.clock() if callable(self.clock) else self.clock
        return d.date() if isinstance(d, datetime) else d

    @property
    def _is_daily_loss_disabled(self) -> bool:
        """En backtest diario (1d), el limite de perdida diaria y circuit breaker quedan apagados."""
        env = getattr(self.portfolio, "environment", None)
        env_val = getattr(env, "value", env)
        is_backtest = env_val == "backtest" or getattr(self.portfolio, "backtest_run_id", None) is not None
        return bool(is_backtest and self.timeframe in ("1d", "1D"))

    def validate_trade(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        stop_price: Optional[float] = None,
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

        # Checks 1 a 4 frenan entradas nuevas: una orden que reduce o cierra la posición
        # existente nunca se rechaza por ellos (QuantAgent-oat, QuantAgent-8wxb).
        held = self.portfolio.positions[symbol]["qty"] if symbol in self.portfolio.positions else 0.0
        opposes = (held > 0 and side == OrderSide.SELL) or (held < 0 and side == OrderSide.BUY)
        reduces = opposes and qty <= abs(held) + self._position_epsilon

        # Check 1: Capital available
        if not reduces and self.portfolio.cash < trade_value:
            reason = f"Insufficient capital: need ${trade_value:.2f}, have ${self.portfolio.cash:.2f}"
            logger.warning(
                f"Order rejected - {reason}",
                extra={
                    "event_type": "order_rejected",
                    "symbol": symbol,
                    "extra_data": {
                        "reason": "insufficient_capital",
                        "trade_value": trade_value,
                        "available_cash": self.portfolio.cash,
                    },
                },
            )
            return (False, reason)

        # Check 2: Position size <= 10% of portfolio
        portfolio_value = self.portfolio.get_total_value()
        max_position_value = portfolio_value * self.max_position_pct
        if not reduces and trade_value > max_position_value:
            reason = f"Position too large: ${trade_value:.2f} > max ${max_position_value:.2f} (10% limit)"
            logger.warning(
                f"Order rejected - {reason}",
                extra={
                    "event_type": "order_rejected",
                    "symbol": symbol,
                    "extra_data": {
                        "reason": "position_limit",
                        "trade_value": trade_value,
                        "max_position_value": max_position_value,
                    },
                },
            )
            return (False, reason)

        # Check 3: Daily loss limit not exceeded
        daily_pnl = self.get_daily_pnl()
        max_daily_loss = -(portfolio_value * self.max_daily_loss_pct)
        if not self._is_daily_loss_disabled and not reduces and daily_pnl < max_daily_loss:
            reason = f"Daily loss limit exceeded: ${daily_pnl:.2f} < max loss ${max_daily_loss:.2f} (5% limit)"
            logger.warning(
                f"Order rejected - {reason}",
                extra={
                    "event_type": "order_rejected",
                    "symbol": symbol,
                    "extra_data": {
                        "reason": "daily_loss_limit",
                        "daily_pnl": daily_pnl,
                        "max_daily_loss": max_daily_loss,
                    },
                },
            )
            return (False, reason)

        # Check 4: Circuit breaker not triggered
        if not self._is_daily_loss_disabled and not reduces and self.circuit_breaker_triggered:
            reason = "Circuit breaker is active - no more trades allowed today"
            logger.warning(
                f"Order rejected - {reason}",
                extra={
                    "event_type": "order_rejected",
                    "symbol": symbol,
                    "extra_data": {"reason": "circuit_breaker"},
                },
            )
            return (False, reason)

        if not reduces and self.max_heat_pct:
            open_risk = sum(
                pos["qty"] * (pos["avg_cost"] - pos["stop_price"])
                for pos in self.portfolio.positions.values()
                if pos.get("qty", 0) > 0
                and pos.get("stop_price") is not None
                and pos["stop_price"] < pos.get("avg_cost", 0)
            )
            new_risk = qty * (price - stop_price) if side == OrderSide.BUY and stop_price is not None and stop_price < price else 0.0
            equity = self.portfolio.get_total_value()
            if (open_risk + new_risk) / equity > self.max_heat_pct:
                reason = f"max_heat: {(open_risk + new_risk) / equity:.2%} > {self.max_heat_pct:.2%}"
                logger.warning(
                    f"Order rejected - {reason}",
                    extra={"event_type": "order_rejected", "symbol": symbol,
                           "extra_data": {"reason": "max_heat", "heat": (open_risk + new_risk) / equity,
                                          "max_heat": self.max_heat_pct}},
                )
                return (False, reason)

        # Check 5: Position management - prevent adding to existing positions
        # (only allow closing or reversing)
        if symbol in self.portfolio.positions:
            existing_pos = self.portfolio.positions[symbol]
            existing_qty = existing_pos["qty"]

            if abs(existing_qty) < self._position_epsilon:
                existing_qty = 0.0

            if existing_qty != 0:
                is_long_position = existing_qty > 0
                is_short_position = existing_qty < 0

                # Prevent adding to LONG position
                if is_long_position and side == OrderSide.BUY:
                    reason = (
                        f"Position already open: LONG {abs(existing_qty):.6f} shares. "
                        f"Cannot add to existing LONG position (prevents over-concentration)"
                    )
                    logger.warning(
                        f"Order rejected - {reason}",
                        extra={
                            "event_type": "order_rejected",
                            "symbol": symbol,
                            "extra_data": {
                                "reason": "add_to_long",
                                "existing_qty": existing_qty,
                            },
                        },
                    )
                    return (False, reason)

                # Prevent adding to SHORT position
                if is_short_position and side == OrderSide.SELL:
                    reason = (
                        f"Position already open: SHORT {abs(existing_qty):.6f} shares. "
                        f"Cannot add to existing SHORT position (prevents over-concentration)"
                    )
                    logger.warning(
                        f"Order rejected - {reason}",
                        extra={
                            "event_type": "order_rejected",
                            "symbol": symbol,
                            "extra_data": {
                                "reason": "add_to_short",
                                "existing_qty": existing_qty,
                            },
                        },
                    )
                    return (False, reason)

                # Allow closing/reversing positions (LONG→SELL, SHORT→BUY)

        return (True, None)

    def get_daily_pnl(self) -> float:
        """
        Get today's realized and unrealized P&L.

        Returns:
            Total P&L for today (sum of realized trades + unrealized positions)
        """
        today = self._get_date()

        # If no DB, use in-memory tracker
        if not self.db:
            return self.daily_pnl_tracker.get(today, 0.0)

        # Query realized trades from today, scoped to the current backtest run
        # (if any) so a prior run's trades don't count against this one's
        # daily loss limit -- see QuantAgent-plan30-D09.
        query = self.db.query(Trade).filter(
            Trade.closed_at >= datetime.combine(today, datetime.min.time()),
            Trade.closed_at <= datetime.combine(today, datetime.max.time()),
        )
        backtest_run_id = getattr(self.portfolio, "backtest_run_id", None)
        if backtest_run_id is not None:
            query = query.filter(Trade.backtest_run_id == backtest_run_id)
        trades_today = query.all()

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
        today = self._get_date()

        # Update daily P&L tracker
        current_daily_pnl = self.daily_pnl_tracker.get(today, 0.0)
        trade_pnl = float(trade.pnl) if trade.pnl else 0.0
        self.daily_pnl_tracker[today] = current_daily_pnl + trade_pnl

        # Check if daily loss limit is now exceeded
        if not self._is_daily_loss_disabled:
            portfolio_value = self.portfolio.get_total_value()
            max_daily_loss = -(portfolio_value * self.max_daily_loss_pct)

            if self.daily_pnl_tracker[today] < max_daily_loss:
                self.circuit_breaker_triggered = True

    def reset_daily_tracker(self, current_date: Optional[date] = None) -> None:
        """Reset daily P&L tracker (call at start of each day)."""
        self.circuit_breaker_triggered = False
        today = current_date or self._get_date()
        self.daily_pnl_tracker[today] = 0.0

    def check_circuit_breaker(self) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit breaker is active.

        Returns:
            Tuple of (is_active, reason_if_active)
        """
        if self._is_daily_loss_disabled:
            return (False, None)
        if self.circuit_breaker_triggered:
            return (
                True,
                "Circuit breaker is active - daily loss limit exceeded",
            )
        return (False, None)
