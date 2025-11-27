"""Portfolio Manager for tracking positions, capital, and P&L."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional
from sqlalchemy.orm import Session

from quantagent.models import Order, Trade, Position, OrderStatus, OrderSide, Environment
from quantagent.database import SessionLocal


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
        db: Optional[Session] = None
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
        self.positions: Dict[str, Dict] = {}  # symbol → {qty, avg_cost, current_price, pnl}

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

        # Get entry price for SELL orders (from position being closed)
        entry_price_for_sell = None
        if order.side == OrderSide.SELL:
            if symbol in self.positions:
                entry_price_for_sell = self.positions[symbol]["avg_cost"]
            else:
                raise ValueError(f"No position in {symbol} to sell")

        # Update positions based on side
        if order.side == OrderSide.BUY:
            self._execute_buy(symbol, fill_qty, fill_price)
        elif order.side == OrderSide.SELL:
            self._execute_sell(symbol, fill_qty, fill_price)

        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= trade_value
        else:
            self.cash += trade_value

        # Create Trade record
        trade = Trade(
            symbol=symbol,
            order_id=order.id,
            entry_price=Decimal(str(fill_price)) if order.side == OrderSide.BUY else Decimal(str(entry_price_for_sell)) if entry_price_for_sell else None,
            exit_price=Decimal(str(fill_price)) if order.side == OrderSide.SELL else None,
            quantity=Decimal(str(fill_qty)),
            side=order.side,
            commission=Decimal(str(0)),  # TODO: Support commission
            environment=self.environment,
            opened_at=datetime.utcnow() if order.side == OrderSide.BUY else None,
            closed_at=datetime.utcnow() if order.side == OrderSide.SELL else None,
        )

        # Persist to database
        self.db.add(trade)
        self._persist_positions()
        self.db.commit()

        return trade

    def _execute_buy(self, symbol: str, qty: float, price: float) -> None:
        """Update position for BUY order."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "qty": 0.0,
                "avg_cost": 0.0,
                "current_price": price,
                "pnl": 0.0,
                "pnl_pct": 0.0,
            }

        pos = self.positions[symbol]
        total_qty = pos["qty"] + qty

        # Calculate new average cost
        if total_qty > 0:
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + qty * price) / total_qty

        pos["qty"] = total_qty
        pos["current_price"] = price
        self._update_position_pnl(symbol)

    def _execute_sell(self, symbol: str, qty: float, price: float) -> None:
        """Update position for SELL order."""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol} to sell")

        pos = self.positions[symbol]
        if pos["qty"] < qty:
            raise ValueError(f"Insufficient qty in {symbol}: have {pos['qty']}, trying to sell {qty}")

        pos["qty"] -= qty
        pos["current_price"] = price

        # If position is fully closed, reset avg_cost
        if pos["qty"] == 0:
            pos["avg_cost"] = 0.0

        self._update_position_pnl(symbol)

    def _update_position_pnl(self, symbol: str) -> None:
        """Calculate unrealized P&L for position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if pos["qty"] == 0:
            pos["pnl"] = 0.0
            pos["pnl_pct"] = 0.0
        else:
            pos["pnl"] = pos["qty"] * (pos["current_price"] - pos["avg_cost"])
            pos["pnl_pct"] = ((pos["current_price"] - pos["avg_cost"]) / pos["avg_cost"]) * 100

    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions).

        Returns:
            Total portfolio value in base currency
        """
        position_value = sum(
            pos["qty"] * pos["current_price"]
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def get_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L.

        Returns:
            Sum of all position P&L
        """
        return sum(
            pos["pnl"] for pos in self.positions.values()
        )

    def get_daily_pnl(self) -> float:
        """Calculate today's realized + unrealized P&L.

        Returns:
            Total P&L for today (realized trades + unrealized positions)
        """
        from datetime import date
        today = date.today()

        # Query trades from today
        trades_today = self.db.query(Trade).filter(
            Trade.closed_at >= datetime.combine(today, datetime.min.time()),
            Trade.environment == self.environment,
        ).all()

        realized_pnl = sum(float(t.pnl) if t.pnl else 0.0 for t in trades_today)
        unrealized_pnl = self.get_unrealized_pnl()

        return realized_pnl + unrealized_pnl

    def get_realized_pnl(self) -> float:
        """Calculate total realized P&L from closed trades.

        Returns:
            Sum of realized P&L from Trade records
        """
        trades = self.db.query(Trade).filter(
            Trade.environment == self.environment,
            Trade.pnl.isnot(None),
            Trade.closed_at.isnot(None)
        ).all()

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
                    side=OrderSide.BUY,  # TODO: Handle short positions
                    opened_at=datetime.utcnow(),
                )
                self.db.add(db_pos)

    def close(self) -> None:
        """Close database session."""
        if self.db:
            self.db.close()
