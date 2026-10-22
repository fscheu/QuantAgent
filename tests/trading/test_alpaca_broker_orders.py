from decimal import Decimal
from types import SimpleNamespace
import pytest
from quantagent.models import Order, OrderSide, OrderStatus, OrderType
from quantagent.trading.alpaca_broker import AlpacaBroker
class Client:
    def __init__(self, result=None, error=None): self.result, self.error = result, error
    def submit_order(self, order_data):
        self.request = order_data
        if self.error: raise self.error
        return self.result
    def cancel_order_by_id(self, order_id): self.cancelled = order_id
def make_order(kind=OrderType.MARKET):
    return Order(symbol="SPY", side=OrderSide.BUY, order_type=kind, quantity=Decimal("2"), price=Decimal("100"))
@pytest.mark.parametrize("kind,expected", [(OrderType.MARKET, "market"), (OrderType.LIMIT, "limit")])
def test_submit_maps_market_and_limit(kind, expected):
    client = Client(SimpleNamespace(status="new", filled_qty="0"))
    order = make_order(kind)
    assert AlpacaBroker(client).place_order(order).status == OrderStatus.PENDING
    assert client.request.type == expected and client.request.qty == "2"
@pytest.mark.parametrize("qty,status", [("2", OrderStatus.FILLED), ("1", OrderStatus.PARTIALLY_FILLED)])
def test_submit_preserves_cumulative_fills(qty, status):
    client = Client(SimpleNamespace(status=status.value, filled_qty=qty, filled_avg_price="101"))
    order = AlpacaBroker(client).submit_order(make_order())
    assert order.status == status and order.filled_quantity == Decimal(qty)
    assert order.fills[0].quantity == Decimal(qty)
def test_rejection_and_cancel():
    broker = AlpacaBroker(Client(SimpleNamespace(status="rejected", reject_reason="insufficient buying power")))
    order = broker.submit_order(make_order())
    assert order.status == OrderStatus.REJECTED and order.comment == "insufficient buying power"
    assert broker.cancel_order("external-id")
    assert broker.client.cancelled == "external-id"
def test_buying_power_api_error_is_rejection():
    error = Exception("40310000 insufficient buying power")
    error.code = 40310000
    order = AlpacaBroker(Client(error=error)).submit_order(make_order())
    assert order.status == OrderStatus.REJECTED and "insufficient buying power" in order.comment
