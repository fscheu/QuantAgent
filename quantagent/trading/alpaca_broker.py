"""Paper Alpaca order adapter using an injected trading client."""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from quantagent.models import Fill, Order, OrderSide, OrderStatus, OrderType
from quantagent.trading.paper_broker import Broker


class AlpacaBroker(Broker):
    def __init__(self, client):
        self.client = client

    def submit_order(self, order: Order) -> Order:
        types = {OrderType.MARKET: "market", OrderType.LIMIT: "limit"}
        if order.order_type not in types:
            raise ValueError(f"Unsupported Alpaca order type: {order.order_type}")
        if order.order_type == OrderType.LIMIT and order.price is None:
            raise ValueError("Limit order requires price")
        fields = dict(symbol=order.symbol, qty=str(order.quantity), side=order.side.value,
                      time_in_force="day")
        if order.order_type == OrderType.LIMIT:
            fields["limit_price"] = str(order.price)
        try:
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
            request = (LimitOrderRequest if order.order_type == OrderType.LIMIT else MarketOrderRequest)(**fields)
        except ImportError:
            request = SimpleNamespace(**fields, type=types[order.order_type])
        try:
            result = self.client.submit_order(order_data=request)
        except Exception as exc:
            if getattr(exc, "code", None) == 40310000:
                order.status, order.comment = OrderStatus.REJECTED, str(exc)
                return order
            raise
        order.alpaca_order_id = str(result.id) if getattr(result, "id", None) else None
        status = str(getattr(result.status, "value", result.status)).lower()
        if status == "rejected":
            order.status = OrderStatus.REJECTED
            order.comment = getattr(result, "reject_reason", None) or getattr(result, "message", None)
            return order
        cumulative = Decimal(str(getattr(result, "filled_qty", 0) or 0))
        previous_filled = Decimal(str(order.filled_quantity or 0))
        if cumulative > previous_filled:
            delta = cumulative - previous_filled
            price = Decimal(str(result.filled_avg_price))
            order.fills.append(Fill(order_id=order.id, quantity=delta, price=price,
                                    commission=Decimal("0"), filled_at=datetime.utcnow()))
            order.filled_quantity = cumulative
            order.average_fill_price = float(price)
        order.status = {"new": OrderStatus.PENDING, "accepted": OrderStatus.PENDING,
                        "partially_filled": OrderStatus.PARTIALLY_FILLED,
                        "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELLED,
                        "rejected": OrderStatus.REJECTED, "expired": OrderStatus.EXPIRED}.get(
                            status, OrderStatus.PENDING)
        if order.status == OrderStatus.FILLED:
            order.filled_at = datetime.utcnow()
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception as exc:
            if getattr(exc, "code", None) == 42210000:
                return False
            raise
        return True

    def get_balance(self):
        return float(self.get_account()["cash"])

    def get_account(self):
        account = self.client.get_account()
        return vars(account)

    def get_positions(self):
        return {position.symbol: position for position in self.client.get_all_positions()}
