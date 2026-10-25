from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from quantagent.fabrica.cartera import curva_cartera, leer_sombra, pesos_objetivo
from quantagent.models import Environment, Order, OrderSide, OrderStatus, OrderType, Trade
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


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_rebalance_fills_are_incremental_and_durable(db_session, side):
    portfolio = PortfolioManager(10000, db=db_session)
    portfolio.positions["SPY"] = {
        "qty": 10 if side == OrderSide.SELL else 0, "avg_cost": 100,
        "current_price": 100, "pnl": 0, "pnl_pct": 0,
    }
    client = SimpleNamespace(get_order_by_id=lambda remote_id: result)
    broker = AlpacaBroker(client)
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                           broker, portfolio, db_session)
    order = Order(symbol="SPY", side=side, quantity=10, order_type=OrderType.MARKET,
                  filled_quantity=0, alpaca_order_id="synthetic-order")
    db_session.add(order)
    db_session.commit()
    @event.listens_for(db_session, "before_commit")
    def assert_fill_is_applied(session):
        for filled in session.query(Order).filter(Order.filled_quantity > 0):
            applied_qty = sum(trade.quantity for trade in session.query(Trade).filter_by(order_id=filled.id))
            assert applied_qty == filled.filled_quantity

    for status, quantity, price in [
        ("accepted", "0", None), ("partially_filled", "3", "101"),
        ("filled", "10", "103"), ("filled", "10", "103"),
    ]:
        result = SimpleNamespace(status=status, filled_qty=quantity, filled_avg_price=price)
        assert manager.apply_rebalance_fills(broker.refresh_order(order)) is order
        db_session.rollback()
        trades = db_session.query(Trade).order_by(Trade.id).all()
        assert [float(trade.quantity) for trade in trades] == (
            [] if quantity == "0" else [3] if quantity == "3" else [3, 7])
        applied = sum(trade.quantity * (trade.exit_price if trade.exit_price is not None
                                       else trade.entry_price) for trade in trades)
        expected = 0 if quantity == "0" else 303 if quantity == "3" else 1030
        assert float(applied) == pytest.approx(expected, rel=0, abs=5e-8)
        direction = 1 if side == OrderSide.SELL else -1
        assert portfolio.cash == pytest.approx(10000 + direction * expected, rel=0, abs=5e-8)
        assert portfolio.positions["SPY"]["qty"] == (10 if direction == 1 else 0) - direction * int(quantity)
        assert order.filled_quantity == Decimal(quantity)
        assert order.average_fill_price == (Decimal(price) if price else None)
    assert float(trades[-1].exit_price if side == OrderSide.SELL else trades[-1].entry_price) == pytest.approx(
        727 / 7, rel=0, abs=5e-9)
    assert db_session.query(Order).count() == 1
    saved_positions = {symbol: position.copy() for symbol, position in portfolio.positions.items()}
    saved_cash, order_id, bind = portfolio.cash, order.id, db_session.get_bind()
    db_session.close()
    with Session(bind) as reopened:
        manager.db = reopened
        portfolio.db = reopened
        order = reopened.get(Order, order_id)
        manager.apply_rebalance_fills(broker.refresh_order(order))
        reopened.rollback()
        assert reopened.query(Trade).count() == 2
        assert reopened.query(Order).count() == 1
        assert portfolio.cash == saved_cash
        assert portfolio.positions == saved_positions


def test_pending_rebalance_guard_and_refresh_lifecycle(db_session):
    portfolio = PortfolioManager(10000, db=db_session)
    remote = SimpleNamespace(status="accepted", filled_qty="0", filled_avg_price=None, reject_reason=None)
    broker = AlpacaBroker(SimpleNamespace(get_order_by_id=lambda _: remote))
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session), broker, portfolio, db_session)
    order = Order(symbol="SPY", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET,
                  environment=Environment.PAPER, alpaca_order_id="alp-1")
    db_session.add(order)
    db_session.commit()

    assert manager.execute_rebalance_order("SPY", OrderSide.BUY, 5, 100) is None

    bind = db_session.get_bind()
    with Session(bind) as reopened:
        reopened_mgr = OrderManager(
            PositionSizer(), RiskManager(portfolio, db=reopened), broker, portfolio, reopened)
        assert reopened_mgr.execute_rebalance_order("SPY", OrderSide.BUY, 5, 100) is None

    remote.status, remote.filled_qty, remote.filled_avg_price = "partially_filled", "3", "101"
    manager.refresh_rebalance_orders()
    assert [float(t.quantity) for t in db_session.query(Trade).all()] == [3]

    manager.refresh_rebalance_orders()
    assert db_session.query(Trade).count() == 1

    remote.status, remote.filled_qty, remote.filled_avg_price = "filled", "10", "103"
    manager.refresh_rebalance_orders()
    assert [float(t.quantity) for t in db_session.query(Trade).order_by(Trade.id)] == [3, 7]
    assert order.status == OrderStatus.FILLED
    assert manager._pending_rebalance_orders().count() == 0


def test_pending_rebalance_terminal_states_and_exclusions(db_session):
    portfolio = PortfolioManager(10000, db=db_session)
    remote = SimpleNamespace(status="partially_filled", filled_qty="3", filled_avg_price="101", reject_reason=None)
    broker = AlpacaBroker(SimpleNamespace(get_order_by_id=lambda _: remote))
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session), broker, portfolio, db_session)

    order_partial = Order(symbol="SPY", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET,
                          environment=Environment.PAPER, alpaca_order_id="alp-partial")
    order_bt = Order(symbol="SPY", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET,
                     environment=Environment.BACKTEST, alpaca_order_id="alp-bt")
    order_prod = Order(symbol="SPY", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET,
                       environment=Environment.PROD, alpaca_order_id="alp-prod")
    db_session.add_all([order_partial, order_bt, order_prod])
    db_session.commit()

    manager.refresh_rebalance_orders()
    assert db_session.query(Trade).count() == 1

    remote.status = "canceled"
    manager.refresh_rebalance_orders()
    assert order_partial.status == OrderStatus.CANCELLED
    assert db_session.query(Trade).count() == 1

    order_rej = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET,
                      environment=Environment.PAPER, alpaca_order_id="alp-rej")
    db_session.add(order_rej)
    db_session.commit()
    remote.status, remote.filled_qty, remote.filled_avg_price = "rejected", "0", None
    remote.reject_reason = "insufficient buying power"
    manager.refresh_rebalance_orders()
    assert order_rej.status == OrderStatus.REJECTED
    assert order_rej.comment == "insufficient buying power"
    assert db_session.query(Trade).filter(Trade.order_id == order_rej.id).count() == 0

    assert manager._pending_rebalance_orders().all() == []

    paper_mgr = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                             PaperBroker(slippage_pct=0), portfolio, db_session)
    order_pending = Order(symbol="MSFT", side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET,
                          environment=Environment.PAPER, alpaca_order_id="alp-pend")
    db_session.add(order_pending)
    db_session.commit()
    assert paper_mgr.execute_rebalance_order("MSFT", OrderSide.BUY, 2, 50) is not None


def test_alpaca_rebalance_buy_reserves_cash_and_truncates_to_integer(db_session):
    client = SimpleNamespace(
        submit_order=lambda order_data: SimpleNamespace(
            id="fake-order-buy", status="filled", filled_qty=order_data.qty,
            filled_avg_price=getattr(order_data, "limit_price", "100.0"), reject_reason=None))
    portfolio = PortfolioManager(1000, db=db_session)
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                           AlpacaBroker(client), portfolio, db_session, cash_reserve_pct=0.05)
    order = manager.execute_rebalance_order("SPY", OrderSide.BUY, 10, 100)
    assert order is not None
    assert order.quantity == 9
    assert order.quantity % 1 == 0
    assert portfolio.cash == 100


@pytest.mark.parametrize("model,kw,expected_qty", [
    ("pct", {"commission_pct": 0.01}, 93),
    ("fixed", {"commission_fixed": 50.0}, 93),
])
def test_paper_rebalance_buy_preserves_costs_and_truncates_integer(db_session, model, kw, expected_qty):
    portfolio = PortfolioManager(10000, db=db_session)
    broker = PaperBroker(slippage_pct=0.01, commission_model=model, **kw)
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                           broker, portfolio, db_session, cash_reserve_pct=0.05)
    order = manager.execute_rebalance_order("SPY", OrderSide.BUY, 100, 100)
    assert order is not None
    assert order.quantity == expected_qty
    assert order.quantity % 1 == 0


@pytest.mark.parametrize("initial_cash,expected_submits,expected_cash", [
    (0, [("sell", 10.0)], 0),
    (100, [("sell", 10.0)], 100),
    (250, [("sell", 10.0), ("buy", 2.0)], 50),
])
def test_rebalance_buy_with_pending_sale_limits_to_available_cash(
    db_session, tmp_path, initial_cash, expected_submits, expected_cash
):
    submitted = []

    def submit(order_data):
        side = getattr(order_data.side, "value", order_data.side)
        submitted.append((side, float(order_data.qty)))
        return SimpleNamespace(
            id=f"fake-{side}", status="accepted" if side == "sell" else "filled",
            filled_qty="0" if side == "sell" else order_data.qty,
            filled_avg_price="100.0", reject_reason=None)

    portfolio = PortfolioManager(initial_cash, db=db_session)
    portfolio.positions["A"] = dict(qty=10, avg_cost=100, current_price=100, pnl=0, pnl_pct=0)
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session),
                           AlpacaBroker(SimpleNamespace(submit_order=submit)), portfolio,
                           db_session, cash_reserve_pct=0.05)
    rebalancear({"A": 0, "B": 0.95}, {"A": 100, "B": 100}, manager,
                tmp_path / "shadow.json", fecha="2026-10-24")
    assert submitted == expected_submits
    assert portfolio.cash == expected_cash
