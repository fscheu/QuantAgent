"""
Paper Broker: Simulates order execution with realistic slippage.

Only receives VALIDATED orders (RiskManager has already approved).
Simulates:
- 2% slippage on fills (±1%)
- Order status transitions (PENDING → FILLED)
- Realistic fill prices
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

from quantagent.models import Order, OrderStatus, OrderSide

logger = logging.getLogger(__name__)


class Broker(ABC):
    """Abstract Broker interface."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place an order and return filled order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Get account balance."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict:
        """Get open positions."""
        pass


class PaperBroker(Broker):
    """Paper (simulated) broker for backtesting and paper trading."""

    def __init__(self, slippage_pct: float = 0.01):
        """
        Initialize Paper Broker.

        Args:
            slippage_pct: Slippage percentage (default 1%, so ±1% for total 2%)
        """
        if not 0 <= slippage_pct <= 0.05:
            raise ValueError("slippage_pct should be between 0% and 5%")
        self.slippage_pct = slippage_pct

    def place_order(self, order: Order) -> Order:
        """
        Place an order and immediately fill it with slippage.

        Args:
            order: Order object with symbol, side, quantity, price

        Returns:
            Filled Order with actual fill_price and filled_quantity

        Note:
            Order already validated by RiskManager, just execute.
        """
        if not order.price:
            raise ValueError("Order price must be set")

        # Simulate realistic fill price with slippage
        if order.side == OrderSide.BUY:
            # BUY: market moves against us slightly
            fill_price = order.price * (1 + self.slippage_pct)
        else:  # SELL
            # SELL: market moves against us slightly
            fill_price = order.price * (1 - self.slippage_pct)

        # Fill entire order quantity
        filled_qty = float(order.quantity)

        # Update order with fill details
        order.filled_price = fill_price
        order.filled_quantity = filled_qty
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()

        logger.info(
            f"{order.symbol}: Order filled - {order.side} {filled_qty:.6f} "
            f"@ ${fill_price:.2f} (slippage: {self.slippage_pct:.2%})"
        )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (MVP: not supported in paper broker)."""
        logger.warning(f"Order cancellation not yet supported in paper broker (order_id={order_id})")
        return False

    def get_balance(self) -> float:
        """Get account balance (would need to query portfolio in real implementation)."""
        raise NotImplementedError("Use portfolio_manager.get_cash() instead")

    def get_positions(self) -> Dict:
        """Get open positions (would need to query portfolio in real implementation)."""
        raise NotImplementedError("Use portfolio_manager.get_positions() instead")
