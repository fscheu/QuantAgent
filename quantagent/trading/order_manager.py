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
from typing import Optional
from sqlalchemy.orm import Session

from quantagent.models import Order, OrderSide, OrderType, Signal, TradeSignal
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
    ):
        """
        Initialize Order Manager.

        Args:
            position_sizer: PositionSizer instance
            risk_manager: RiskManager instance
            broker: Broker instance (PaperBroker)
            portfolio_manager: PortfolioManager instance
            db: SQLAlchemy session
        """
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.broker = broker
        self.portfolio = portfolio_manager
        self.db = db

    def execute_decision(
        self,
        symbol: str,
        decision: TradeSignal,
        confidence: float,
        current_price: float,
        environment=None,
        trigger_signal_id: Optional[int] = None,
    ) -> Optional[Order]:
        """
        Execute a trading decision end-to-end.

        Flow:
        1. If HOLD → return None
        2. Calculate size based on confidence
        3. Validate trade
        4. If invalid → log rejection, return None
        5. Create Order object
        6. Place with broker
        7. Update portfolio
        8. Update risk tracker
        9. Log to database

        Args:
            symbol: Trading symbol (e.g., "BTC", "SPX")
            decision: "LONG", "SHORT", or "HOLD"
            confidence: Signal confidence (0-1)
            current_price: Current market price
            environment: Environment enum (BACKTEST, PAPER, PROD) - optional
            trigger_signal_id: ID of signal that triggered this order - optional

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
            )

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
                filled_order, filled_order.average_fill_price
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

    def _execute_reversal(
        self,
        symbol: str,
        existing_qty: float,
        new_side: OrderSide,
        new_qty: float,
        current_price: float,
        environment=None,
        trigger_signal_id: Optional[int] = None,
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

        Returns:
            Filled Order for the new position if successful, None if failed
        """
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
                filled_close_order, filled_close_order.average_fill_price
            )
            logger.info(
                f"{symbol}: Portfolio updated - close {close_side} {close_qty:.6f} executed"
            )
        except Exception as e:
            logger.error(f"{symbol}: Close order portfolio update failed - {str(e)}")
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
            f"{symbol}: Executing reversal - Step 2: Open {new_side} {new_qty:.6f}"
        )

        # Validate new trade
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, new_side, new_qty, current_price
        )
        if not is_valid:
            logger.warning(
                f"{symbol}: New position order rejected - {reason} (position already closed)"
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
                filled_new_order, filled_new_order.average_fill_price
            )
            logger.info(
                f"{symbol}: Portfolio updated - new {new_side} {new_qty:.6f} executed"
            )
        except Exception as e:
            logger.error(f"{symbol}: New order portfolio update failed - {str(e)}")
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

        logger.info(f"{symbol}: Position reversal completed successfully")
        return filled_new_order
