from types import SimpleNamespace
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.models import OrderSide
from quantagent.trading.bucket_m5 import (MAX_HEAT_PCT, MAX_POSITION_PCT,
    RIESGO_POR_TRADE_PCT, cargar_credenciales, crear_position_sizer, crear_risk_manager)
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.stop_management import posiciones_con_stops, resolver_stop_posicion
from quantagent.database import Base
from quantagent.trading.position_monitor import PositionMonitor


class Portfolio:
    def __init__(self, positions, equity):
        self.positions = positions
        self.cash = equity
        self.equity = equity

    def get_total_value(self):
        return self.equity


def test_m5_profile_heat_and_sizing_leave_m4_default_unchanged(monkeypatch):
    monkeypatch.delenv("TRADING_MAX_HEAT_PCT", raising=False)
    import quantagent.settings as settings
    monkeypatch.setattr(settings, "TRADING_MAX_HEAT_PCT", 0)
    risk = crear_risk_manager(Portfolio({}, equity=3000))
    assert risk.max_heat_pct == MAX_HEAT_PCT == 0.02
    assert risk.max_position_pct == MAX_POSITION_PCT == 0.10
    assert crear_position_sizer().riesgo_por_trade_pct == RIESGO_POR_TRADE_PCT == 0.0025
    assert RiskManager(Portfolio({}, equity=3000)).max_heat_pct == 0


def test_two_stopped_positions_reject_new_trade_on_total_heat():
    portfolio = Portfolio({
        "A": {"qty": 10, "avg_cost": 100, "stop_price": 90},
        "B": {"qty": 10, "avg_cost": 100, "stop_price": 80},
    }, equity=10_000)
    risk = crear_risk_manager(portfolio)
    accepted, reason = risk.validate_trade("C", OrderSide.BUY, 10, 100, stop_price=90)
    assert not accepted
    assert "max_heat" in reason


def test_stop_resolution_prefers_open_broker_stop():
    order = SimpleNamespace(symbol="SPY", type="stop", side="sell", stop_price=91.5)
    intent = SimpleNamespace(intent={"stop_loss": 88})
    assert resolver_stop_posicion("SPY", [order], [intent]) == 91.5
    assert resolver_stop_posicion("SPY", [], [intent]) == 88
    assert resolver_stop_posicion("SPY", [], []) is None


def test_missing_stop_is_omitted_and_does_not_add_heat():
    portfolio = Portfolio({"OLD": {"qty": 10, "avg_cost": 100}}, equity=10_000)
    risk = crear_risk_manager(portfolio)
    assert risk.validate_trade("NEW", OrderSide.BUY, 10, 100, stop_price=90)[0]
    remote = {"OLD": SimpleNamespace(qty="10", avg_entry_price="100",
                                     current_price="100", unrealized_pl="0")}
    assert "stop_price" not in posiciones_con_stops(remote, [], [])["OLD"]


def test_credentials_complete_both_alpaca_name_sets(tmp_path, monkeypatch):
    env_file = tmp_path / "alpaca.env"
    env_file.write_text("APCA_API_KEY_ID=key-one\nAPCA_API_SECRET_KEY=secret-one\n")
    cargar_credenciales(env_file)
    assert os.environ["ALPACA_API_KEY"] == "key-one"
    assert os.environ["ALPACA_SECRET_KEY"] == "secret-one"


def test_reconcile_uses_only_the_configured_account_positions(tmp_path):
    engines = [create_engine(f"sqlite:///{tmp_path / name}.db") for name in ("m5", "m4")]
    for engine in engines:
        Base.metadata.create_all(engine)
    with Session(engines[0]) as m5_db, Session(engines[1]) as m4_db:
        m5_broker = SimpleNamespace(get_positions=lambda: {
            "SPY": SimpleNamespace(qty="1", side="long")})
        m4_broker = SimpleNamespace(get_positions=lambda: {
            "QQQ": SimpleNamespace(qty="2", side="long")})
        m5_diff = PositionMonitor(m5_db).reconcile_alpaca_positions(m5_broker)
        m4_diff = PositionMonitor(m4_db).reconcile_alpaca_positions(m4_broker)
        assert m5_diff == ["SPY: remote-only qty=1 side=long"]
        assert m4_diff == ["QQQ: remote-only qty=2 side=long"]
    for engine in engines:
        engine.dispose()
