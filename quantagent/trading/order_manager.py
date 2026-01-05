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

        # Step 3: Detect position reversal
        existing_position = self.portfolio.positions.get(symbol)
        is_reversal = False
        if existing_position:
            existing_qty = existing_position["qty"]
            is_reversal = (existing_qty > 0 and side == OrderSide.SELL) or (
                existing_qty < 0 and side == OrderSide.BUY
            )

        # Step 4: Execute reversal or single order
        if is_reversal:
            logger.info(
                f"{symbol}: Position reversal detected - closing existing position and opening new one"
            )
            return self._execute_reversal(
                symbol=symbol,
                new_side=side,
                new_qty=qty,
                current_price=current_price,
                environment=environment,
                trigger_signal_id=trigger_signal_id,
            )

        # Step 5: Validate trade (now includes position management check)
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

    def _execute_reversal(
        self,
        symbol: str,
        new_side: OrderSide,
        new_qty: float,
        current_price: float,
        environment=None,
        trigger_signal_id: Optional[int] = None,
    ) -> Optional[Order]:
        """
        Execute a position reversal as two separate orders.

        Args:
            symbol: Trading symbol
            new_side: Desired new position side (BUY for LONG, SELL for SHORT)
            new_qty: Desired new position quantity
            current_price: Current market price
            environment: Environment enum
            trigger_signal_id: ID of signal that triggered this reversal

        Returns:
            Last filled Order (open order) if successful, None if failed
        """
        existing_position = self.portfolio.positions[symbol]
        existing_qty = existing_position["qty"]

        # Step 1: Close existing position
        close_side = OrderSide.BUY if existing_qty < 0 else OrderSide.SELL
        close_qty = abs(existing_qty)

        logger.info(
            f"{symbol}: Reversal step 1/2 - closing existing position: "
            f"{close_side} {close_qty:.6f} @ ${current_price:.2f}"
        )

        # Validate close order
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, close_side, close_qty, current_price
        )
        if not is_valid:
            logger.error(
                f"{symbol}: Reversal failed - close order rejected: {reason}"
            )
            return None

        # Create and execute close order
        close_order = Order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            price=current_price,
            order_type=OrderType.MARKET,
            environment=environment,
            trigger_signal_id=trigger_signal_id,
        )

        try:
            self.db.add(close_order)
            self.db.flush()
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal failed - close order persistence failed: {str(e)}"
            )
            self.db.rollback()
            return None

        try:
            filled_close_order = self.broker.place_order(close_order)
            logger.info(
                f"{symbol}: Close order filled - {filled_close_order.side} "
                f"{filled_close_order.filled_quantity:.6f} @ "
                f"${filled_close_order.average_fill_price:.2f}"
            )
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal failed - close order broker execution failed: {str(e)}"
            )
            return None

        try:
            close_trade = self.portfolio.execute_trade(
                filled_close_order, filled_close_order.average_fill_price
            )
            self.risk_manager.on_trade_executed(close_trade)
            self.db.add(close_trade)
            self.db.flush()
            logger.info(f"{symbol}: Close position completed")
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal failed - close portfolio update failed: {str(e)}"
            )
            return None

        # Step 2: Open new position
        logger.info(
            f"{symbol}: Reversal step 2/2 - opening new position: "
            f"{new_side} {new_qty:.6f} @ ${current_price:.2f}"
        )

        # Validate open order
        is_valid, reason = self.risk_manager.validate_trade(
            symbol, new_side, new_qty, current_price
        )
        if not is_valid:
            logger.error(
                f"{symbol}: Reversal partial - open order rejected: {reason} "
                f"(existing position closed successfully)"
            )
            return None

        # Create and execute open order
        open_order = Order(
            symbol=symbol,
            side=new_side,
            quantity=new_qty,
            price=current_price,
            order_type=OrderType.MARKET,
            environment=environment,
            trigger_signal_id=trigger_signal_id,
        )

        try:
            self.db.add(open_order)
            self.db.flush()
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal partial - open order persistence failed: {str(e)}"
            )
            self.db.rollback()
            return None

        try:
            filled_open_order = self.broker.place_order(open_order)
            logger.info(
                f"{symbol}: Open order filled - {filled_open_order.side} "
                f"{filled_open_order.filled_quantity:.6f} @ "
                f"${filled_open_order.average_fill_price:.2f}"
            )
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal partial - open order broker execution failed: {str(e)}"
            )
            return None

        try:
            open_trade = self.portfolio.execute_trade(
                filled_open_order, filled_open_order.average_fill_price
            )
            self.risk_manager.on_trade_executed(open_trade)
            self.db.add(open_trade)
            self.db.commit()
            logger.info(f"{symbol}: Position reversal completed successfully")
        except Exception as e:
            logger.error(
                f"{symbol}: Reversal partial - open portfolio update failed: {str(e)}"
            )
            self.db.rollback()
            return None

        # Update reverse provenance link if available
        if trigger_signal_id:
            try:
                sig = (
                    self.db.query(Signal)
                    .filter(Signal.id == trigger_signal_id)
                    .first()
                )
                if sig and not sig.order_id:
                    sig.order_id = open_order.id
                    self.db.commit()
            except Exception as e:
                logger.warning(
                    f"{symbol}: Failed to update signal provenance: {str(e)}"
                )

        return filled_open_order

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
