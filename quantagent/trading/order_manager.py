"""
Order Manager: Orchestrates the complete order execution flow.

Flow:
1. PositionSizer.calculate_size() → qty
2. RiskManager.validate_trade() → (valid, reason)
   - If False: REJECT (return None)
   - If True: continue
3. Create Order object
4. PaperBroker.place_order() → filled_order
5. PortfolioManager.execute_trade() → Trade
6. RiskManager.on_trade_executed() → update daily P&L
7. Database.add(trade) → commit
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy.orm import Session

from quantagent.models import Environment, Order, OrderSide, OrderStatus, OrderType, Signal, Trade, TradeSignal

from .alpaca_broker import AlpacaBroker
from .paper_broker import PaperBroker
from .position_sizer import PositionSizer
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderManager:
    """Orchestrates order execution: Size → Validate → Execute → Update → Log."""

    def __init__(
        self,
        position_sizer: PositionSizer,
        risk_manager: RiskManager,
        broker,  # PaperBroker instance
        portfolio_manager,  # PortfolioManager instance
        db: Session,
        orden_minima: Optional[float] = None,
        cash_reserve_pct: float = 0.0,
    ):
        """
        Initialize Order Manager.

        Args:
            position_sizer: PositionSizer instance
            risk_manager: RiskManager instance
            broker: Broker instance (PaperBroker)
            portfolio_manager: PortfolioManager instance
            db: SQLAlchemy session
            orden_minima: Minimum order value in USD for entries (optional)
        """
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.broker = broker
        self.portfolio = portfolio_manager
        self.db = db
        self.orden_minima = (
            orden_minima
            if orden_minima is not None
            else getattr(position_sizer, "orden_minima", None)
        )
        self.entradas_descartadas_orden_minima = 0
        self.cash_reserve_pct = cash_reserve_pct

    def _cap_entry_qty(self, side: OrderSide, qty: float, price: float) -> float:
        """Limita compras al efectivo disponible después de la reserva y los costos."""
        cash = self.portfolio.cash
        if side != OrderSide.BUY or qty * price > cash * (1 + 1e-9):
            return qty
        cash *= 1 - self.cash_reserve_pct
        unit_cost = price
        if isinstance(self.broker, PaperBroker):
            unit_cost = price * (1 + self.broker.slippage_pct)
            if self.broker.commission_model == "pct":
                unit_cost *= 1 + self.broker.commission_pct
            elif self.broker.commission_model == "fixed":
                cash -= self.broker.commission_fixed
        max_qty = cash / unit_cost
        if qty <= max_qty or max_qty <= 0:
            return qty
        logger.info(f"Entry qty capped to cash: {qty:.6f} -> {max_qty:.6f}")
        return max_qty

    def execute_decision(
        self,
        symbol: str,
        decision: TradeSignal,
        confidence: float,
        current_price: float,
        environment=None,
        trigger_signal_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Execute a trading decision end-to-end.

        Flow:
        1. If HOLD → return None
        2. Calculate size based on confidence
        3. Detect position reversal (SHORT->LONG or LONG->SHORT)
        4. If reversal: execute two-order reversal (close + open)
        5. If not reversal: validate trade and execute single order
        6. Update risk tracker
        7. Log to database

        Args:
            symbol: Trading symbol (e.g., "BTC", "SPX")
            decision: "LONG", "SHORT", or "HOLD"
            confidence: Signal confidence (0-1)
            current_price: Current market price
            environment: Environment enum (BACKTEST, PAPER, PROD) - optional
            trigger_signal_id: ID of signal that triggered this order - optional
            timestamp: Simulated candle time for backtests; None (wall-clock) for
                live/paper trading
            stop_price: Signal stop used by the risk-based position sizing rule

        Returns:
            Filled Order if executed, None if rejected
        """
        # Step 1: HOLD decision
        if decision == TradeSignal.NEUTRAL:
            logger.info(f"{symbol}: NEUTRAL signal, no trade executed")
            return None

        # Step 2: Calculate position size
        portfolio_value = self.portfolio.get_total_value()
        qty = self.position_sizer.calculate_size(
            symbol=symbol,
            signal_confidence=confidence,
            current_price=current_price,
            portfolio_value=portfolio_value,
            stop_price=stop_price,
        )

        logger.info(
            f"{symbol}: Size calculated - {decision} {qty:.6f} @ ${current_price:.2f} "
            f"(confidence={confidence:.1%}, portfolio=${portfolio_value:.2f})"
        )

        # Determine order side BEFORE validation (needed for position check)
        side = OrderSide.BUY if decision.upper() == "LONG" else OrderSide.SELL

        # Check for position reversal
        current_position = self.portfolio.get_position(symbol)
        existing_qty = current_position.get("qty", 0.0) if current_position else 0.0

        is_reversal = (existing_qty > 0 and side == OrderSide.SELL) or (
            existing_qty < 0 and side == OrderSide.BUY
        )

        if is_reversal:
            logger.info(
                f"{symbol}: Position reversal detected - existing qty: {existing_qty}, new side: {side}"
            )
            # Execute reversal as two orders: close existing + open new
            return self._execute_reversal(
                symbol=symbol,
                existing_qty=existing_qty,
                new_side=side,
                new_qty=qty,
                current_price=current_price,
                environment=environment,
                trigger_signal_id=trigger_signal_id,
                timestamp=timestamp,
            )

        if self.orden_minima is not None and (qty <= 0 or (qty * current_price) < self.orden_minima - 1e-9):
            self.entradas_descartadas_orden_minima += 1
            logger.info(
                f"{symbol}: Entry discarded due to min order size {self.orden_minima} USD "
                f"(qty: {qty:.6f}, value: ${qty * current_price:.2f})"
            )
            return None

        qty = self._cap_entry_qty(side, qty, current_price)

        # Step 3: Validate trade (now includes position management check)
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, side, qty, current_price
        )

        if not is_valid:
            logger.warning(f"{symbol}: Trade rejected - {reason}")
            return None

        logger.info(f"{symbol}: Trade validation passed - proceeding to execution")

        # Step 4: Create Order object
        order = Order(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=current_price,
            order_type=OrderType.MARKET,
            environment=environment,  # Set environment (backtest/paper/prod)
            trigger_signal_id=trigger_signal_id,  # Set provenance link
        )

        # Persist order early to obtain ID for provenance
        try:
            self.db.add(order)
            self.db.flush()
        except Exception as e:
            logger.error(f"{symbol}: Failed to persist order pre-fill - {str(e)}")
            self.db.rollback()
            return None

        # Step 5: Place with broker
        try:
            filled_order = self.broker.place_order(order)
            logger.info(
                f"{symbol}: Order filled - {filled_order.side} {filled_order.filled_quantity:.6f} "
                f"@ ${filled_order.average_fill_price:.2f}"
            )
        except Exception as e:
            logger.error(f"{symbol}: Broker execution failed - {str(e)}")
            return None

        # Step 6: Update portfolio
        try:
            trade = self.portfolio.execute_trade(
                filled_order, filled_order.average_fill_price, timestamp=timestamp
            )
            logger.info(f"{symbol}: Portfolio updated - {side} {qty:.6f} executed")
        except Exception as e:
            logger.error(f"{symbol}: Portfolio update failed - {str(e)}")
            return None

        # Step 7: Update risk tracker (post-trade P&L)
        self.risk_manager.on_trade_executed(trade)

        # Step 8: Log to database (persist filled order and trade)
        try:
            # Update reverse provenance link if available
            if trigger_signal_id:
                sig = (
                    self.db.query(Signal).filter(Signal.id == trigger_signal_id).first()
                )
                if sig and not sig.order_id:
                    sig.order_id = order.id
            self.db.add(trade)
            self.db.commit()
            logger.info(f"{symbol}: Trade logged to database")
        except Exception as e:
            logger.error(f"{symbol}: Database logging failed - {str(e)}")
            self.db.rollback()
            return None

        return filled_order

    def execute_decision_with_order(
        self,
        order: Order,
        decision: str,
        confidence: float,
        current_price: float,
    ) -> Optional[Order]:
        """
        Execute an order that's already created (alternative to execute_decision).

        Args:
            order: Pre-created Order object
            decision: "LONG", "SHORT", or "HOLD"
            confidence: Signal confidence
            current_price: Current market price

        Returns:
            Filled Order if executed, None if rejected
        """
        if decision.upper() == "HOLD":
            logger.info(f"{order.symbol}: HOLD signal, no trade executed")
            return None

        # Validate trade (includes position management check)
        is_valid, reason = self.risk_manager.validate_trade(
            order.symbol,
            order.side,
            order.quantity,
            order.price or current_price,
        )

        if not is_valid:
            logger.warning(f"{order.symbol}: Trade rejected - {reason}")
            return None

        # Place with broker
        try:
            filled_order = self.broker.place_order(order)
        except Exception as e:
            logger.error(f"{order.symbol}: Broker execution failed - {str(e)}")
            return None

        # Update portfolio
        try:
            trade = self.portfolio.execute_trade(
                filled_order, filled_order.average_fill_price
            )
        except Exception as e:
            logger.error(f"{order.symbol}: Portfolio update failed - {str(e)}")
            return None

        # Update risk tracker
        self.risk_manager.on_trade_executed(trade)

        return filled_order

    def execute_rebalance_order(self, symbol: str, side: OrderSide, qty: float, price: float) -> Optional[Order]:
        """Execute a portfolio rebalance without strategy risk or per-entry caps."""
        if (isinstance(self.broker, AlpacaBroker) and self.portfolio.environment == Environment.PAPER
                and self._pending_rebalance_orders().filter(Order.symbol == symbol).first()):
            return None
        position = self.portfolio.get_position(symbol)
        held = float(position.get("qty", 0)) if position else 0.0
        if side == OrderSide.SELL:
            qty = min(qty, max(0.0, held))
        else:
            qty = int(self._cap_entry_qty(side, min(qty, max(0.0, self.portfolio.cash) / price), price))
        if qty <= 0 or (self.orden_minima is not None and qty * price < self.orden_minima - 1e-9):
            return None
        order = Order(symbol=symbol, side=side, quantity=qty, price=price, order_type=OrderType.MARKET,
                      environment=self.portfolio.environment)
        try:
            self.db.add(order)
            self.db.flush()
            filled = self.broker.place_order(order)
            return self.apply_rebalance_fills(filled)
        except Exception as exc:
            logger.error("%s: Rebalance order failed - %s", symbol, exc)
            self.db.rollback()
            return None

    def _pending_rebalance_orders(self):
        return self.db.query(Order).filter(
            Order.environment == Environment.PAPER, Order.alpaca_order_id.isnot(None),
            Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]))

    def refresh_rebalance_orders(self):
        """Refresh persisted paper Alpaca orders and apply confirmed fill deltas."""
        if not isinstance(self.broker, AlpacaBroker) or self.portfolio.environment != Environment.PAPER:
            return
        for order in self._pending_rebalance_orders().all():
            self.apply_rebalance_fills(self.broker.refresh_order(order))

    def apply_rebalance_fills(self, order: Order) -> Order:
        """Apply only remote fills not yet represented by this order's Trades."""
        if not order.filled_quantity:
            self.db.commit()
            return order
        execution, price = order, order.average_fill_price
        if order.alpaca_order_id:
            trades = self.db.query(Trade).filter(Trade.order_id == order.id).all()
            applied_qty = sum((trade.quantity for trade in trades), Decimal("0"))
            delta = Decimal(str(order.filled_quantity)) - applied_qty
            if delta <= 0:
                self.db.commit()
                return order
            applied_notional = sum(
                (trade.quantity * (trade.exit_price if trade.exit_price is not None else trade.entry_price)
                 for trade in trades), Decimal("0"))
            remote_notional = Decimal(str(order.filled_quantity)) * Decimal(str(order.average_fill_price))
            price = float((remote_notional - applied_notional) / delta)
            execution = Order(id=order.id, symbol=order.symbol, side=order.side,
                              quantity=delta, filled_quantity=delta)
        trade = self.portfolio.execute_trade(execution, price)
        self.risk_manager.on_trade_executed(trade)
        return order

    def _execute_reversal(
        self,
        symbol: str,
        existing_qty: float,
        new_side: OrderSide,
        new_qty: float,
        current_price: float,
        environment=None,
        trigger_signal_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Order]:
        """
        Execute position reversal as two orders: close existing position, then open new.

        Args:
            symbol: Trading symbol
            existing_qty: Current position quantity (positive=LONG, negative=SHORT)
            new_side: Side of new position (BUY=LONG, SELL=SHORT)
            new_qty: Quantity for new position
            current_price: Current market price
            environment: Environment enum
            trigger_signal_id: ID of signal that triggered this order
            timestamp: Simulated candle time for backtests; None (wall-clock) for
                live/paper trading

        Returns:
            Filled Order for the new position if successful, None if failed
        """
        # Log initial state
        portfolio_value = self.portfolio.get_total_value()
        logger.info(
            f"{symbol}: Starting position reversal - "
            f"Current: {('LONG' if existing_qty > 0 else 'SHORT')} {abs(existing_qty):.6f}, "
            f"Target: {new_side.name} {new_qty:.6f}, "
            f"Portfolio value: ${portfolio_value:.2f}"
        )

        # Step 1: Close existing position
        close_side = OrderSide.SELL if existing_qty > 0 else OrderSide.BUY
        close_qty = abs(existing_qty)

        logger.info(
            f"{symbol}: Executing reversal - Step 1: Close {close_side} {close_qty:.6f}"
        )

        # Validate close trade
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, close_side, close_qty, current_price
        )
        if not is_valid:
            logger.warning(f"{symbol}: Close order rejected - {reason}")
            return None

        # Create close order
        close_order = Order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            price=current_price,
            order_type=OrderType.MARKET,
            environment=environment,
            trigger_signal_id=trigger_signal_id,
        )

        # Persist close order
        try:
            self.db.add(close_order)
            self.db.flush()
        except Exception as e:
            logger.error(f"{symbol}: Failed to persist close order - {str(e)}")
            self.db.rollback()
            return None

        # Execute close order with broker
        try:
            filled_close_order = self.broker.place_order(close_order)
            logger.info(
                f"{symbol}: Close order filled - {filled_close_order.side} {filled_close_order.filled_quantity:.6f} "
                f"@ ${filled_close_order.average_fill_price:.2f}"
            )
        except Exception as e:
            logger.error(f"{symbol}: Close order broker execution failed - {str(e)}")
            return None

        # Update portfolio for close
        try:
            close_trade = self.portfolio.execute_trade(
                filled_close_order,
                filled_close_order.average_fill_price,
                timestamp=timestamp,
            )
            logger.info(
                f"{symbol}: Portfolio updated - close {close_side} {close_qty:.6f} executed"
            )
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal ABORTED at step 1 - Close order portfolio update failed: {type(e).__name__}: {str(e)}. "
                f"Position may remain in original state. Portfolio value: ${self.portfolio.get_total_value():.2f}"
            )
            return None

        # Update risk tracker for close
        self.risk_manager.on_trade_executed(close_trade)

        # Persist close trade
        try:
            if trigger_signal_id:
                sig = (
                    self.db.query(Signal)
                    .filter(Signal.id == trigger_signal_id)
                    .first()
                )
                if sig and not sig.order_id:
                    sig.order_id = close_order.id
            self.db.add(close_trade)
            self.db.commit()
            logger.info(f"{symbol}: Close trade logged to database")
        except Exception as e:
            logger.error(f"{symbol}: Close trade database logging failed - {str(e)}")
            self.db.rollback()
            return None

        # Step 2: Open new position
        logger.info(
            f"{symbol}: Reversal step 1/2 complete - position closed. "
            f"Proceeding to step 2: Open {new_side} {new_qty:.6f}"
        )

        # Validate new trade (the opening leg is an entry: cap it to the cash left after the close)
        if self.orden_minima is not None and (new_qty <= 0 or (new_qty * current_price) < self.orden_minima - 1e-9):
            self.entradas_descartadas_orden_minima += 1
            logger.info(
                f"{symbol}: Reversal new entry discarded due to min order size {self.orden_minima} USD "
                f"(qty: {new_qty:.6f}, value: ${new_qty * current_price:.2f})"
            )
            return None

        new_qty = self._cap_entry_qty(new_side, new_qty, current_price)
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, new_side, new_qty, current_price
        )
        if not is_valid:
            logger.error(
                f"{symbol}: Reversal PARTIAL FAILURE - Close succeeded but new position rejected: {reason}. "
                f"Position is now FLAT (no exposure). Portfolio value: ${self.portfolio.get_total_value():.2f}"
            )
            return None

        # Create new position order
        new_order = Order(
            symbol=symbol,
            side=new_side,
            quantity=new_qty,
            price=current_price,
            order_type=OrderType.MARKET,
            environment=environment,
            trigger_signal_id=trigger_signal_id,
        )

        # Persist new order
        try:
            self.db.add(new_order)
            self.db.flush()
        except Exception as e:
            logger.error(f"{symbol}: Failed to persist new order - {str(e)}")
            self.db.rollback()
            return None

        # Execute new order with broker
        try:
            filled_new_order = self.broker.place_order(new_order)
            logger.info(
                f"{symbol}: New order filled - {filled_new_order.side} {filled_new_order.filled_quantity:.6f} "
                f"@ ${filled_new_order.average_fill_price:.2f}"
            )
        except Exception as e:
            logger.error(f"{symbol}: New order broker execution failed - {str(e)}")
            return None

        # Update portfolio for new position
        try:
            new_trade = self.portfolio.execute_trade(
                filled_new_order,
                filled_new_order.average_fill_price,
                timestamp=timestamp,
            )
            logger.info(
                f"{symbol}: Portfolio updated - new {new_side} {new_qty:.6f} executed"
            )
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal PARTIAL FAILURE - Close succeeded but new order portfolio update failed: {type(e).__name__}: {str(e)}. "
                f"Position is now FLAT (no exposure). Portfolio value: ${self.portfolio.get_total_value():.2f}"
            )
            return None

        # Update risk tracker for new position
        self.risk_manager.on_trade_executed(new_trade)

        # Persist new trade
        try:
            self.db.add(new_trade)
            self.db.commit()
            logger.info(f"{symbol}: New trade logged to database")
        except Exception as e:
            logger.error(f"{symbol}: New trade database logging failed - {str(e)}")
            self.db.rollback()
            return None

        logger.info(
            f"{symbol}: Position reversal COMPLETE - "
            f"New position: {new_side} {self.portfolio.positions.get(symbol, {}).get('qty', 0):.6f} "
            f"@ ${self.portfolio.positions.get(symbol, {}).get('avg_cost', 0):.2f}. "
            f"Portfolio value: ${self.portfolio.get_total_value():.2f}"
        )
        return filled_new_order

    def reset_daily_tracker(self, current_date: Optional[date] = None) -> None:
        """Delegate daily-tracker reset to RiskManager (facade method)."""
        self.risk_manager.reset_daily_tracker(current_date)

    def close_trade(
        self,
        trade_id: int,
        current_price: float,
        environment=None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Order]:
        """Close an open trade by executing an opposing order at current_price.

        Args:
            timestamp: Simulated candle time for backtests; None (wall-clock) for
                live/paper trading
        """
        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if trade is None:
            logger.warning(f"close_trade: trade_id={trade_id} not found")
            return None

        close_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY

        current_position = self.portfolio.positions.get(trade.symbol, {})
        current_qty = abs(float(current_position.get("qty", 0.0)))
        if current_qty <= 0:
            logger.warning(f"{trade.symbol}: close_trade skipped - no portfolio position to close")
            return None

        qty = min(float(trade.quantity), current_qty)
        close_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
        return self._close_position(trade.symbol, qty, current_price, close_side, environment, timestamp)

    def close_symbol(
        self,
        symbol: str,
        current_price: float,
        environment=None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Order]:
        """Close the complete portfolio position for a symbol."""
        current_position = self.portfolio.positions.get(symbol, {})
        qty = abs(float(current_position.get("qty", 0.0)))
        if qty <= 0:
            logger.warning(f"{symbol}: close_symbol skipped - no portfolio position to close")
            return None
        close_side = OrderSide.SELL if float(current_position.get("qty", 0.0)) > 0 else OrderSide.BUY
        return self._close_position(symbol, qty, current_price, close_side, environment, timestamp)

    def _close_position(
        self,
        symbol: str,
        qty: float,
        current_price: float,
        close_side: OrderSide,
        environment=None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Order]:
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, close_side, qty, current_price
        )
        if not is_valid:
            logger.warning(f"{symbol}: close rejected - {reason}")
            return None

        order = Order(
            symbol=symbol,
            side=close_side,
            quantity=qty,
            price=current_price,
            order_type=OrderType.MARKET,
            environment=environment,
        )

        try:
            self.db.add(order)
            self.db.flush()
        except Exception as e:
            logger.error(f"{symbol}: close failed to persist order - {e}")
            self.db.rollback()
            return None

        try:
            filled_order = self.broker.place_order(order)
        except Exception as e:
            logger.error(f"{symbol}: close broker execution failed - {e}")
            return None

        try:
            close_trade_record = self.portfolio.execute_trade(
                filled_order, filled_order.average_fill_price, timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"{symbol}: close portfolio update failed - {e}")
            return None

        self.risk_manager.on_trade_executed(close_trade_record)

        try:
            self.db.add(close_trade_record)
            self.db.commit()
        except Exception as e:
            logger.error(f"{symbol}: close database commit failed - {e}")
            self.db.rollback()
            return None

        return filled_order
