import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from quantagent.database import Base
from quantagent.models import Order, OrderSide, OrderStatus, OrderType
from quantagent.trading.alpaca_broker import AlpacaBroker


class Client:
    def __init__(self, result=None, error=None, account=None, positions=None):
        self.result, self.error = result, error
        self.account, self.positions = account, positions
    def submit_order(self, order_data):
        self.request = order_data
        if self.error:
            raise self.error
        return self.result
    def cancel_order_by_id(self, order_id):
        self.cancelled = order_id
        if self.error:
            raise self.error
    def get_account(self):
        if self.error:
            raise self.error
        return self.account
    def get_all_positions(self):
        if self.error:
            raise self.error
        return self.positions
def make_order(kind=OrderType.MARKET):
    return Order(symbol="SPY", side=OrderSide.BUY, order_type=kind, quantity=Decimal("2"), price=Decimal("100"))
@pytest.mark.parametrize("kind,expected", [(OrderType.MARKET, "market"), (OrderType.LIMIT, "limit")])
def test_submit_maps_market_and_limit(kind, expected):
    client = Client(SimpleNamespace(id="alpaca-123", status="new", filled_qty="0"))
    order = make_order(kind)
    assert AlpacaBroker(client).place_order(order).status == OrderStatus.PENDING
    assert str(getattr(client.request.type, "value", client.request.type)).lower() == expected
    assert float(client.request.qty) == 2
    assert order.alpaca_order_id == "alpaca-123"
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

@pytest.mark.parametrize("status,expected", [("new", OrderStatus.PENDING), ("accepted", OrderStatus.PENDING),
    ("partially_filled", OrderStatus.PARTIALLY_FILLED), ("filled", OrderStatus.FILLED),
    ("canceled", OrderStatus.CANCELLED), ("rejected", OrderStatus.REJECTED), ("expired", OrderStatus.EXPIRED)])
def test_submit_maps_alpaca_statuses(status, expected):
    result = SimpleNamespace(status=status, filled_qty="0")
    assert AlpacaBroker(Client(result)).submit_order(make_order()).status == expected

def test_account_balance_positions_and_api_errors():
    account = SimpleNamespace(cash="1234.50", buying_power="2000")
    positions = [SimpleNamespace(symbol="SPY", qty="2")]
    broker = AlpacaBroker(Client(account=account, positions=positions))
    assert broker.get_account() == vars(account)
    assert broker.get_balance() == 1234.5
    assert broker.get_positions() == {"SPY": positions[0]}
    error = RuntimeError("API unavailable")
    broker = AlpacaBroker(Client(error=error))
    with pytest.raises(RuntimeError, match="API unavailable"):
        broker.get_account()
    with pytest.raises(RuntimeError, match="API unavailable"):
        broker.get_positions()
    with pytest.raises(RuntimeError, match="API unavailable"):
        broker.cancel_order("id")

def test_cancel_returns_false_when_alpaca_rejects_cancellation():
    error = Exception("order cannot be canceled")
    error.code = 42210000
    assert not AlpacaBroker(Client(error=error)).cancel_order("id")

def test_remote_order_id_survives_new_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'orders.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    order = make_order()
    AlpacaBroker(Client(SimpleNamespace(id="alpaca-durable", status="new", filled_qty="0"))).submit_order(order)
    with Session() as session:
        session.add(order)
        session.commit()
        order_id = order.id
    with Session() as session:
        assert session.get(Order, order_id).alpaca_order_id == "alpaca-durable"
    engine.dispose()

def test_alpaca_order_id_migration_round_trip(tmp_path):
    db_path = tmp_path / "migration.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE orders (id INTEGER PRIMARY KEY, symbol TEXT)"))
        connection.execute(text("INSERT INTO orders VALUES (1, 'SPY')"))
    engine.dispose()
    env = dict(os.environ, DATABASE_URL=f"sqlite:///{db_path}")
    alembic = [sys.executable, "-m", "alembic"]
    root = Path(__file__).resolve().parents[2]
    steps = [("stamp", "4c2a1d8e7f90", False), ("upgrade", "7d2e4a91c6b3", True),
             ("downgrade", "4c2a1d8e7f90", False), ("upgrade", "7d2e4a91c6b3", True)]
    for command, revision, has_column in steps:
        args = (command, revision)
        subprocess.run([*alembic, *args], cwd=root, env=env, check=True, capture_output=True)
        engine = create_engine(f"sqlite:///{db_path}")
        assert ("alpaca_order_id" in {column["name"] for column in inspect(engine).get_columns("orders")}) == has_column
        with engine.connect() as connection:
            assert connection.execute(text("SELECT id, symbol FROM orders")).one() == (1, "SPY")
        engine.dispose()
