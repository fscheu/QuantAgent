from types import SimpleNamespace

import pandas as pd
import pytest

from quantagent.fabrica.cartera import curva_cartera, leer_sombra, pesos_objetivo
from quantagent.models import Order, OrderSide, OrderStatus, Trade
from quantagent.portfolio.manager import PortfolioManager
from quantagent.trading.alpaca_broker import AlpacaBroker
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.rebalanceador import ordenes_rebalanceo, rebalancear
from quantagent.trading.risk_manager import RiskManager


def test_t4_target_weights_are_rebalanced_and_saved_to_new_shadow(db_session, tmp_path):
    fecha = pd.Timestamp("2026-10-01")
    piernas = {"transversal": ["A", "B"], "base": ["C"]}
    cierres = pd.DataFrame([[100, 100, 100]], index=[fecha], columns=["A", "B", "C"])
    panel = pd.DataFrame([[1, 0]], index=[fecha], columns=["A", "B"])
    seleccion = {"transversal": panel}
    objetivos = pesos_objetivo(cierres, piernas, fecha, seleccion=seleccion)
    assert objetivos == pytest.approx({"A": .5, "B": 0, "C": .5})

    retornos = {(pierna, activo): pd.Series([.1 if activo != "B" else 0], index=[fecha])
                for pierna, activos in piernas.items() for activo in activos}
    assert curva_cartera(cierres, piernas, retornos, seleccion=seleccion).iloc[-1] == pytest.approx(110000)

    portfolio = PortfolioManager(100000, db=db_session)
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                           PaperBroker(slippage_pct=0), portfolio, db_session)
    sombra = tmp_path / "t4-shadow.json"
    rebalancear(objetivos, {activo: 100 for activo in cierres.columns}, manager,
                sombra, fecha=fecha.date())
    assert leer_sombra(sombra, "paper", fecha.date()) == {"pesos": objetivos, "valor": 100000}


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
