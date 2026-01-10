"""Portfolio Manager for tracking positions, capital, and P&L."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from sqlalchemy.orm import Session

from quantagent.database import SessionLocal
from quantagent.models import (
    Environment,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    Trade,
)


class PortfolioManager:
    """Manages portfolio state including positions, cash, and P&L calculations.

    Attributes:
        positions: Dict mapping symbol to position data {qty, avg_cost, current_price, pnl}
        cash: Available capital
        environment: Execution environment (backtest, paper, prod)
        total_value: Total portfolio value (cash + position values)
    """

    def __init__(
        self,
        initial_cash: float,
        environment: Environment = Environment.PAPER,
        db: Optional[Session] = None,
    ):
        """Initialize portfolio manager.

        Args:
            initial_cash: Starting capital (e.g., 100000.0)
            environment: Execution environment tag
            db: SQLAlchemy session (uses SessionLocal if not provided)
        """
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.environment = environment
        self.db = db or SessionLocal()
        self.positions: Dict[str, Dict] = (
            {}
        )  # symbol → {qty, avg_cost, current_price, pnl}

    def execute_trade(self, order: Order, fill_price: float) -> Trade:
        """Execute a trade, update positions, persist to database.

        Args:
            order: Order object with symbol, side, quantity
            fill_price: Actual fill price from broker

        Returns:
            Trade object with updated portfolio state

        Raises:
            ValueError: If invalid trade data

        Note:
            Pre-trade validation (capital, position size, daily loss) is handled by RiskManager
            BEFORE this method is called. This method only updates state.
        """
        symbol = order.symbol
        qty = float(order.quantity)
        fill_qty = float(order.filled_quantity) if order.filled_quantity else qty
        trade_value = fill_qty * fill_price

        # NOTE: Capital validation already done by RiskManager.validate_trade()
        # If we reach here, the trade is already validated

        # Determine entry price and action type BEFORE updating position
        entry_price_for_sell = None
        position_qty_before = (
            self.positions[symbol]["qty"] if symbol in self.positions else 0.0
        )

        if order.side == OrderSide.SELL:
            if position_qty_before > 0:
                # Closing LONG position
                entry_price_for_sell = self.positions[symbol]["avg_cost"]
            elif position_qty_before == 0:
                # Opening new SHORT position
                entry_price_for_sell = None
            elif position_qty_before < 0:
                # Increasing existing SHORT position
                entry_price_for_sell = self.positions[symbol]["avg_cost"]
        elif order.side == OrderSide.BUY:
            if position_qty_before < 0:
                # Closing SHORT position
                entry_price_for_sell = self.positions[symbol]["avg_cost"]

        # Update positions based on side
        if order.side == OrderSide.BUY:
            self._execute_buy(symbol, fill_qty, fill_price)
        elif order.side == OrderSide.SELL:
            self._execute_sell(symbol, fill_qty, fill_price)

        # Update cash based on action (determined by position state BEFORE trade)
        if order.side == OrderSide.BUY:
            # BUY always reduces cash (opening LONG or closing SHORT)
            self.cash -= trade_value
        else:  # SELL
            # SELL always increases cash (closing LONG or opening SHORT)
            self.cash += trade_value

        # Create Trade record with correct entry/exit prices based on action type
        # Determine if opening or closing position based on qty BEFORE trade
        is_opening = position_qty_before == 0
        is_closing_long = position_qty_before > 0 and order.side == OrderSide.SELL
        is_closing_short = position_qty_before < 0 and order.side == OrderSide.BUY

        if is_opening:
            # Opening new position (LONG or SHORT)
            entry_price = Decimal(str(fill_price))
            exit_price = None
            opened_at = datetime.utcnow()
            closed_at = None
        elif is_closing_long or is_closing_short:
            # Closing existing position
            entry_price = Decimal(str(entry_price_for_sell))
            exit_price = Decimal(str(fill_price))
            opened_at = (
                None  # Should reference original trade, but we don't track that yet
            )
            closed_at = datetime.utcnow()
        else:
            # Increasing existing position (LONG or SHORT)
            entry_price = Decimal(str(fill_price))
            exit_price = None
            opened_at = datetime.utcnow()
            closed_at = None

        # Calculate P&L for closing trades
        pnl: Decimal | None = None
        pnl_pct: float | None = None

        if is_closing_long or is_closing_short:
            if entry_price and entry_price > 0 and exit_price is not None:
                if is_closing_long:
                    # LONG: profit when exit > entry
                    pnl = (exit_price - entry_price) * Decimal(str(fill_qty))
                    pnl_pct = float((exit_price - entry_price) / entry_price * 100)
                else:  # is_closing_short
                    # SHORT: profit when entry > exit
                    pnl = (entry_price - exit_price) * Decimal(str(fill_qty))
                    pnl_pct = float((entry_price - exit_price) / entry_price * 100)
            else:
                # Edge case: invalid entry_price or exit_price
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Cannot calculate P&L for {symbol}: "
                    f"invalid entry_price={entry_price} or exit_price={exit_price}"
                )

        trade = Trade(
            symbol=symbol,
            order_id=order.id,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=Decimal(str(fill_qty)),
            side=order.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=Decimal(str(0)),  # TODO: Support commission
            environment=self.environment,
            opened_at=opened_at or datetime.utcnow(),  # Ensure opened_at is never None
            closed_at=closed_at,
        )

        # Persist to database
        self.db.add(trade)
        self._persist_positions()
        self.db.commit()

        return trade

    def _execute_buy(self, symbol: str, qty: float, price: float) -> None:
        """Update position for BUY order (open LONG or close SHORT)."""
        if symbol not in self.positions:
            # Open new LONG position
            self.positions[symbol] = {
                "qty": qty,
                "avg_cost": price,
                "current_price": price,
                "pnl": 0.0,
                "pnl_pct": 0.0,
            }
        else:
            pos = self.positions[symbol]
            if pos["qty"] < 0:
                # Close SHORT position
                if abs(pos["qty"]) < qty:
                    raise ValueError(
                        f"Trying to buy {qty} shares but SHORT position in {symbol} is only {abs(pos['qty'])}"
                    )
                pos["qty"] += qty  # Reduce negative (move toward zero)
            else:
                # Increase LONG position
                total_qty = pos["qty"] + qty
                pos["avg_cost"] = (
                    pos["qty"] * pos["avg_cost"] + qty * price
                ) / total_qty
                pos["qty"] = total_qty

            pos["current_price"] = price

            # If position is fully closed, reset avg_cost
            if pos["qty"] == 0:
                pos["avg_cost"] = 0.0

        self._update_position_pnl(symbol)

    def _execute_sell(self, symbol: str, qty: float, price: float) -> None:
        """Update position for SELL order (close LONG or open SHORT)."""
        if symbol not in self.positions:
            # Open new SHORT position (negative qty)
            self.positions[symbol] = {
                "qty": -qty,
                "avg_cost": price,
                "current_price": price,
                "pnl": 0.0,
                "pnl_pct": 0.0,
            }
        else:
            pos = self.positions[symbol]
            if pos["qty"] > 0:
                # Close LONG position
                if pos["qty"] < qty:
                    raise ValueError(
                        f"Insufficient qty in {symbol}: have {pos['qty']}, trying to sell {qty}"
                    )
                pos["qty"] -= qty
            else:
                # Increase SHORT position (make more negative)
                total_qty = pos["qty"] - qty
                pos["avg_cost"] = (
                    abs(pos["qty"]) * pos["avg_cost"] + qty * price
                ) / abs(total_qty)
                pos["qty"] = total_qty

            pos["current_price"] = price

            # If position is fully closed, reset avg_cost
            if pos["qty"] == 0:
                pos["avg_cost"] = 0.0

        self._update_position_pnl(symbol)

    def _update_position_pnl(self, symbol: str) -> None:
        """Calculate unrealized P&L (works for LONG and SHORT positions)."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if pos["qty"] == 0:
            pos["pnl"] = 0.0
            pos["pnl_pct"] = 0.0
        else:
            if pos["qty"] > 0:
                # LONG: profit when price increases
                pos["pnl"] = pos["qty"] * (pos["current_price"] - pos["avg_cost"])
                pos["pnl_pct"] = (
                    (pos["current_price"] - pos["avg_cost"]) / pos["avg_cost"]
                ) * 100
            else:
                # SHORT: profit when price decreases (inverse P&L)
                pos["pnl"] = abs(pos["qty"]) * (pos["avg_cost"] - pos["current_price"])
                pos["pnl_pct"] = (
                    (pos["avg_cost"] - pos["current_price"]) / pos["avg_cost"]
                ) * 100

    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions).

        For LONG: value = qty × current_price
        For SHORT: value = qty × (2 × avg_cost - current_price)
                  = initial_sale_proceeds - current_liability

        Returns:
            Total portfolio value in base currency
        """
        position_value = 0.0
        for pos in self.positions.values():
            if pos["qty"] > 0:
                # LONG: value = shares × price
                position_value += pos["qty"] * pos["current_price"]
            else:
                # SHORT: value = initial proceeds - current buyback cost
                # = qty × avg_cost (proceeds) - qty × current_price (liability)
                # = qty × (2 × avg_cost - current_price)
                position_value += abs(pos["qty"]) * (
                    2 * pos["avg_cost"] - pos["current_price"]
                )

        return self.cash + position_value

    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L.

        Returns:
            Sum of all position P&L
        """
        return sum(pos["pnl"] for pos in self.positions.values())

    def get_daily_pnl(self) -> float:
        """Calculate today's realized + unrealized P&L.

        Returns:
            Total P&L for today (realized trades + unrealized positions)
        """
        from datetime import date

        today = date.today()

        # Query trades from today
        trades_today = (
            self.db.query(Trade)
            .filter(
                Trade.closed_at >= datetime.combine(today, datetime.min.time()),
                Trade.environment == self.environment,
            )
            .all()
        )

        realized_pnl = sum(float(t.pnl) if t.pnl else 0.0 for t in trades_today)
        unrealized_pnl = self.get_unrealized_pnl()

        return realized_pnl + unrealized_pnl

    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed trades.

        Returns:
            Sum of realized P&L from Trade records
        """
        trades = (
            self.db.query(Trade)
            .filter(
                Trade.environment == self.environment,
                Trade.pnl.isnot(None),
                Trade.closed_at.isnot(None),
            )
            .all()
        )

        return float(sum(float(t.pnl) for t in trades))

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for positions.

        Args:
            prices: Dict mapping symbol to current price
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol]["current_price"] = price
                self._update_position_pnl(symbol)

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position dict or None if no position
        """
        return self.positions.get(symbol)

    def get_positions(self) -> Dict[str, Dict]:
        """Get all positions.

        Returns:
            Dict of all positions
        """
        return self.positions.copy()

    def get_cash(self) -> float:
        """Get available cash.

        Returns:
            Available cash amount
        """
        return self.cash

    def _persist_positions(self) -> None:
        """Persist all positions to database.

        This is called after execute_trade to ensure database
        is always in sync with in-memory state.
        """
        for symbol, pos_data in self.positions.items():
            # Check if position exists
            db_pos = self.db.query(Position).filter(Position.symbol == symbol).first()

            if db_pos:
                # Update existing position
                db_pos.quantity = Decimal(str(pos_data["qty"]))
                db_pos.average_entry_price = Decimal(str(pos_data["avg_cost"]))
                db_pos.current_price = Decimal(str(pos_data["current_price"]))
                db_pos.unrealized_pnl = Decimal(str(pos_data["pnl"]))
                db_pos.unrealized_pnl_pct = pos_data["pnl_pct"]
                db_pos.side = OrderSide.BUY if pos_data["qty"] >= 0 else OrderSide.SELL
                db_pos.updated_at = datetime.utcnow()
            else:
                # Create new position
                db_pos = Position(
                    symbol=symbol,
                    quantity=Decimal(str(pos_data["qty"])),
                    average_entry_price=Decimal(str(pos_data["avg_cost"])),
                    current_price=Decimal(str(pos_data["current_price"])),
                    unrealized_pnl=Decimal(str(pos_data["pnl"])),
                    unrealized_pnl_pct=pos_data["pnl_pct"],
                    side=OrderSide.BUY if pos_data["qty"] >= 0 else OrderSide.SELL,
                    opened_at=datetime.utcnow(),
                )
                self.db.add(db_pos)

    def close(self) -> None:
        """Close database session."""
        if self.db:
            self.db.close()
