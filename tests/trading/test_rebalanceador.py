from types import SimpleNamespace

import pytest

from quantagent.models import Order, OrderSide, OrderStatus, Trade
from quantagent.portfolio.manager import PortfolioManager
from quantagent.trading.alpaca_broker import AlpacaBroker
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.rebalanceador import ordenes_rebalanceo, rebalancear
from quantagent.trading.risk_manager import RiskManager


def test_rebalance_existing_positions_sells_before_buying(db_session, tmp_path):
    portfolio = PortfolioManager(2000, db=db_session)
    portfolio.positions = {"A": {"qty": 10, "avg_cost": 100, "current_price": 100, "pnl": 0, "pnl_pct": 0},
                           "B": {"qty": 70, "avg_cost": 100, "current_price": 100, "pnl": 0, "pnl_pct": 0}}
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session), PaperBroker(slippage_pct=0), portfolio, db_session, cash_reserve_pct=.05)
    orders = rebalancear({"A": .45, "B": .10, "C": .40}, {"A": 100, "B": 100, "C": 100}, manager, tmp_path / "shadow.json", fecha="2026-10-22")
    from quantagent.fabrica.cartera import leer_sombra
    assert leer_sombra(tmp_path / "shadow.json", "paper", "2026-10-22") == {"pesos": {"A": .45, "B": .10, "C": .40}, "valor": 10000.0}
    assert [o.side for o in orders if o] == [OrderSide.SELL, OrderSide.BUY, OrderSide.BUY]
    assert {s: p["qty"] for s, p in portfolio.positions.items()} == {"A": 45, "B": 10, "C": 40}


def test_already_at_weights_creates_no_orders():
    adjustments = ordenes_rebalanceo({"A": .5, "B": .5}, {"A": {"qty": 5}, "B": {"qty": 5}}, {"A": 100, "B": 100}, 1000)
    assert adjustments == []


@pytest.mark.parametrize("status,filled_qty,fill_price,expected_status,expected_qty", [
    ("accepted", "0", None, OrderStatus.PENDING, 10),
    ("rejected", "0", None, OrderStatus.REJECTED, 10),
    ("partially_filled", "3", "101", OrderStatus.PARTIALLY_FILLED, 7),
])
def test_alpaca_rebalance_applies_only_real_fills(
    db_session, status, filled_qty, fill_price, expected_status, expected_qty
):
    class Client:
        def submit_order(self, order_data):
            return SimpleNamespace(id="fake-order", status=status,
                                   filled_qty=filled_qty, filled_avg_price=fill_price,
                                   reject_reason="insufficient buying power")

    portfolio = PortfolioManager(10000, db=db_session)
    portfolio.cash = 9000
    portfolio.positions["SPY"] = {
        "qty": 10, "avg_cost": 100, "current_price": 100, "pnl": 0
    }
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                           AlpacaBroker(Client()), portfolio, db_session)

    returned = manager.execute_rebalance_order("SPY", OrderSide.SELL, 10, 100)

    db_session.rollback()
    persisted = db_session.query(Order).one()
    assert returned is not None
    assert persisted.status == expected_status
    assert persisted.comment == ("insufficient buying power" if status == "rejected" else None)
    assert float(persisted.filled_quantity) == (3 if filled_qty == "3" else 0)
    assert db_session.query(Trade).count() == (1 if filled_qty == "3" else 0)
    assert portfolio.positions["SPY"]["qty"] == expected_qty
    assert portfolio.cash == (9303 if filled_qty == "3" else 9000)
