from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from quantagent.database import Base

from quantagent.models import Environment, Order, OrderSide, OrderStatus, OrderType
from quantagent.fabrica.cartera import leer_sombra
from quantagent.settings import SchedulerSettings
from quantagent.portfolio.manager import PortfolioManager
from quantagent.trading.alpaca_broker import AlpacaBroker
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.scheduler import TradingScheduler


def _scheduler(db, manager, portfolio, env="paper"):
    return TradingScheduler(trading_graph=Mock(), order_manager=manager, data_provider=Mock(),
                           db_session=db, scheduler_settings=SchedulerSettings(assets=["SPY"], environment=env),
                           paper_portfolio=portfolio)


def test_run_once_refreshes_pending_before_rebalance_and_skips_missing_panel(db_session, monkeypatch, tmp_path):
    import quantagent.trading.scheduler as module
    from types import SimpleNamespace

    fecha = pd.Timestamp("2026-10-23")
    portfolio = PortfolioManager(10000, db=db_session)
    held = SimpleNamespace(symbol="SPY", qty="2", avg_entry_price="90", current_price="100", unrealized_pl="20")
    client = SimpleNamespace(get_account=lambda: SimpleNamespace(cash=10000), get_all_positions=lambda: [held],
                             get_order_by_id=lambda _: SimpleNamespace(status="accepted", filled_qty="0"))
    broker = AlpacaBroker(client)
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session), broker, portfolio, db_session)
    db_session.add(Order(symbol="SPY", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET,
                         environment=Environment.PAPER, alpaca_order_id="pending"))
    db_session.commit()
    monkeypatch.setattr(module, "entradas_cartera", lambda *a: (fecha, "panel", {"SPY": 100}))
    monkeypatch.setattr(module, "pesos_operables", lambda hipotesis, *a: pd.Series(
        {"SPY": 1.0 if hipotesis == "h.yaml" else 0.1}))
    scheduler = _scheduler(db_session, manager, {"hipotesis": "h.yaml", "reloj": lambda: fecha,
                           "sombra": tmp_path / "sombra.json", "cartera": "test"})

    scheduler.run_once()
    scheduler.run_once()
    assert db_session.query(Order).count() == 1
    assert portfolio.positions["SPY"]["qty"] == 2
    assert leer_sombra(tmp_path / "sombra.json", "test", fecha.date())["pesos"] == {"SPY": 1.0}
    monkeypatch.setattr(module, "entradas_cartera", lambda *a: None)
    assert scheduler.run_once()["errors"] == 0
    assert db_session.query(Order).count() == 1

@pytest.mark.parametrize("environment", ["prod", "backtest"])
def test_paper_portfolio_rejects_non_paper_environment(db_session, environment):
    with pytest.raises(ValueError, match="paper portfolios require paper environment"):
        _scheduler(db_session, Mock(), {}, environment)


def test_two_paper_portfolios_keep_fills_and_shadows_isolated(monkeypatch, tmp_path):
    import quantagent.trading.scheduler as module
    from types import SimpleNamespace
    from quantagent.models import Trade

    monkeypatch.setattr(module, "AlpacaBroker", type("ExternalAccountBroker", (), {}))
    fecha = pd.Timestamp("2026-10-23")
    monkeypatch.setattr(module, "entradas_cartera", lambda *a: (fecha, "panel", {"SPY": 100}))
    monkeypatch.setattr(module, "pesos_operables", lambda *a: pd.Series({"SPY": 0.1}))
    systems = []
    for name in ("nucleo", "transversal"):
        engine = create_engine(f"sqlite:///{tmp_path / (name + '.db')}")
        Base.metadata.create_all(engine)
        db = Session(engine)
        portfolio = PortfolioManager(10000, db=db)
        remote = SimpleNamespace(status="accepted", filled_qty="0", filled_avg_price=None)
        client = SimpleNamespace(
            submit_order=lambda order_data, r=remote, n=name: SimpleNamespace(id=n, status=r.status,
                filled_qty=r.filled_qty, filled_avg_price=r.filled_avg_price),
            get_order_by_id=lambda _, r=remote: r,
            get_account=lambda: SimpleNamespace(cash=10000), get_all_positions=lambda: [])
        broker = AlpacaBroker(client)
        manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db), broker, portfolio, db)
        config = {"hipotesis": name, "reloj": lambda: fecha,
                  "sombra": tmp_path / f"{name}.json", "cartera": name}
        systems.append((engine, db, portfolio, manager, config, remote))

    for _, db, _, manager, config, _ in systems:
        _scheduler(db, manager, config)._process_paper_portfolio()
    for _, db, portfolio, manager, config, _ in systems:
        assert db.query(Order).one().status == OrderStatus.PENDING
        assert portfolio.positions == {}
    for _, db, portfolio, manager, config, remote in systems:
        remote.status, remote.filled_qty, remote.filled_avg_price = "partially_filled", "3", "100"
        _scheduler(db, manager, config)._process_paper_portfolio()
        assert db.query(Order).one().status == OrderStatus.PARTIALLY_FILLED
        assert db.query(Trade).one().quantity == 3
    for index, (engine, db, portfolio, manager, config, remote) in enumerate(systems):
        db.close()
        reopened = Session(engine)
        portfolio.db = manager.db = manager.risk_manager.db = reopened
        remote.status, remote.filled_qty = "filled", "10"
        _scheduler(reopened, manager, config)._process_paper_portfolio()
        assert reopened.query(Order).count() == 1
        assert reopened.query(Trade).count() == 2
        assert portfolio.positions["SPY"]["qty"] == 10
        assert leer_sombra(config["sombra"], config["cartera"], fecha.date())["pesos"] == {"SPY": 0.1}
        assert reopened.query(Order).one().status == OrderStatus.FILLED
        assert reopened.query(Order).one().alpaca_order_id == ("nucleo" if index == 0 else "transversal")
        reopened.close()

