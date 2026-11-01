"""Paper Alpaca order adapter using an injected trading client."""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from quantagent.models import Fill, Order, OrderStatus, OrderType
from quantagent.trading.paper_broker import Broker


class AlpacaBroker(Broker):
    def __init__(self, client):
        self.client = client

    def es_fraccionable(self, simbolo: str) -> bool:
        return bool(self.client.get_asset(simbolo).fractionable)

    def submit_order(self, order: Order) -> Order:
        types = {OrderType.MARKET: "market", OrderType.LIMIT: "limit", OrderType.STOP: "stop"}
        if order.order_type not in types:
            raise ValueError(f"Unsupported Alpaca order type: {order.order_type}")
        if order.order_type == OrderType.LIMIT and order.price is None:
            raise ValueError("Limit order requires price")
        if order.order_type == OrderType.STOP and order.stop_price is None:
            raise ValueError("Stop order requires stop_price")
        tif = ("gtc" if order.quantity == order.quantity.to_integral_value() else "day") \
            if order.order_type == OrderType.STOP else "day"
        fields = dict(symbol=order.symbol, qty=str(order.quantity), side=order.side.value,
                      time_in_force=tif)
        if order.order_type == OrderType.LIMIT:
            fields["limit_price"] = str(order.price)
        elif order.order_type == OrderType.STOP:
            fields["stop_price"] = str(order.stop_price)
        try:
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopOrderRequest
            request_type = {OrderType.LIMIT: LimitOrderRequest, OrderType.MARKET: MarketOrderRequest,
                            OrderType.STOP: StopOrderRequest}[order.order_type]
            request = request_type(**fields)
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
        return self._update_order(order, result)

    def replace_stop(self, alpaca_order_id: str, new_stop_price) -> Order:
        fields = dict(stop_price=new_stop_price)
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            request = ReplaceOrderRequest(**fields)
        except ImportError:
            request = SimpleNamespace(**fields)
        return self.client.replace_order_by_id(alpaca_order_id, request)

    def refresh_order(self, order: Order) -> Order:
        return self._update_order(order, self.client.get_order_by_id(order.alpaca_order_id))

    def _update_order(self, order: Order, result) -> Order:
        status = str(getattr(result.status, "value", result.status)).lower()
        if status == "rejected":
            order.comment = getattr(result, "reject_reason", None) or getattr(result, "message", None)
        cumulative = Decimal(str(getattr(result, "filled_qty", 0) or 0))
        previous_filled = Decimal(str(order.filled_quantity or 0))
        if cumulative > previous_filled:
            delta = cumulative - previous_filled
            price = Decimal(str(result.filled_avg_price))
            previous_notional = previous_filled * Decimal(str(order.average_fill_price or 0))
            delta_price = (cumulative * price - previous_notional) / delta
            order.fills.append(Fill(order_id=order.id, quantity=delta, price=delta_price,
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
