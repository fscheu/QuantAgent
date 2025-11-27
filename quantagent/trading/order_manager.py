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

from quantagent.models import Order, OrderSide, OrderType
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
        decision: str,
        confidence: float,
        current_price: float,
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

        Returns:
            Filled Order if executed, None if rejected
        """
        # Step 1: HOLD decision
        if decision.upper() == "HOLD":
            logger.info(f"{symbol}: HOLD signal, no trade executed")
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

        # Step 3: Validate trade
        is_valid, reason = self.risk_manager.validate_trade(symbol, qty, current_price)

        if not is_valid:
            logger.warning(f"{symbol}: Trade rejected - {reason}")
            return None

        logger.info(f"{symbol}: Trade validation passed - proceeding to execution")

        # Step 4: Create Order object
        side = OrderSide.BUY if decision.upper() == "LONG" else OrderSide.SELL
        order = Order(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=current_price,
            order_type=OrderType.MARKET,
        )

        # Step 5: Place with broker
        try:
            filled_order = self.broker.place_order(order)
            logger.info(
                f"{symbol}: Order filled - {filled_order.side} {filled_order.filled_quantity:.6f} "
                f"@ ${filled_order.filled_price:.2f}"
            )
        except Exception as e:
            logger.error(f"{symbol}: Broker execution failed - {str(e)}")
            return None

        # Step 6: Update portfolio
        try:
            trade = self.portfolio.execute_trade(filled_order, filled_order.filled_price)
            logger.info(f"{symbol}: Portfolio updated - {side} {qty:.6f} executed")
        except Exception as e:
            logger.error(f"{symbol}: Portfolio update failed - {str(e)}")
            return None

        # Step 7: Update risk tracker (post-trade P&L)
        self.risk_manager.on_trade_executed(trade)

        # Step 8: Log to database (already done in portfolio.execute_trade)
        logger.info(f"{symbol}: Trade logged to database")

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

        # Validate trade
        is_valid, reason = self.risk_manager.validate_trade(
            order.symbol,
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
            trade = self.portfolio.execute_trade(filled_order, filled_order.filled_price)
        except Exception as e:
            logger.error(f"{order.symbol}: Portfolio update failed - {str(e)}")
            return None

        # Update risk tracker
        self.risk_manager.on_trade_executed(trade)

        return filled_order
