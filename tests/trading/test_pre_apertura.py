from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.database import Base
from quantagent.models import ActivePosition, Environment, ExitPolicy, OrderSide
from quantagent.trading.alpaca_broker import AlpacaBroker
from quantagent.trading.pre_apertura import correr_pre_apertura


class Client:
    def __init__(self, positions=(), orders=(), daytrade_count=0):
        self.positions, self.orders = list(positions), list(orders)
        self.daytrade_count = daytrade_count
        self.submitted, self.cancelled = [], []

    def get_all_positions(self):
        return self.positions

    def get_orders(self, filter):
        return list(self.orders)

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        order = SimpleNamespace(id=f"stop-{len(self.submitted)}", symbol=order_data.symbol,
                                qty=order_data.qty, side="sell", type="stop", status="new")
        self.orders.append(order)
        return SimpleNamespace(id=order.id, status="new", filled_qty="0")

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(order_id)
        self.orders[:] = [o for o in self.orders if o.id != order_id]

    def get_account(self):
        return SimpleNamespace(daytrade_count=self.daytrade_count)


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pre.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def add_position(session, symbol="SPY", qty="1.25", stop="95"):
    session.add(ActivePosition(symbol=symbol, side=OrderSide.BUY, entry_price=100,
        stop_loss=Decimal(stop), take_profit=110, quantity=Decimal(qty),
        decision_timestamp=datetime(2026, 1, 1), exit_policy=ExitPolicy.SL_TP_ONLY,
        environment=Environment.PAPER, candles_direction=[]))
    session.commit()


def test_fractional_position_gets_day_stop_at_local_stop(tmp_path):
    session, engine = make_session(tmp_path)
    add_position(session)
    client = Client([SimpleNamespace(symbol="SPY", qty="1.25", side="long")])
    report = correr_pre_apertura(session, AlpacaBroker(client), datetime(2026, 10, 28, 12))
    assert report["reenviados"] == ["SPY"]
    assert client.submitted[0].time_in_force == "day"
    assert client.submitted[0].stop_price == 95
    assert report["sin_stop"] == []
    session.close(); engine.dispose()


def test_existing_stop_prevents_duplicate_and_covers_position(tmp_path):
    session, engine = make_session(tmp_path)
    add_position(session)
    stop = SimpleNamespace(id="s1", symbol="SPY", qty="1.25", side="sell", type="stop")
    client = Client([SimpleNamespace(symbol="SPY", qty="1.25", side="long")], [stop])
    report = correr_pre_apertura(session, AlpacaBroker(client), datetime(2026, 10, 28, 12))
    assert not client.submitted and report["sin_stop"] == []
    session.close(); engine.dispose()


def test_integer_position_without_stop_is_reported(tmp_path):
    session, engine = make_session(tmp_path)
    client = Client([SimpleNamespace(symbol="SPY", qty="2", side="long")])
    report = correr_pre_apertura(session, AlpacaBroker(client), datetime(2026, 10, 28, 12))
    assert report["sin_stop"] == ["SPY"]
    session.close(); engine.dispose()


def test_cuenta_nueva_sin_daytrade_count_no_rompe(tmp_path):
    session, engine = make_session(tmp_path)
    report = correr_pre_apertura(session, AlpacaBroker(Client(daytrade_count=None)), datetime(2026, 10, 28, 12))
    assert report["avisos"] == [] and report["sin_stop"] == []
    session.close(); engine.dispose()


def test_old_entry_cancelled_recent_entry_kept_and_pdt_notice(tmp_path):
    session, engine = make_session(tmp_path)
    now = datetime(2026, 10, 28, 12)
    old = SimpleNamespace(id="old", symbol="SPY", qty="1", side="buy", type="stop",
                          created_at=now - timedelta(days=20))
    recent = SimpleNamespace(id="recent", symbol="QQQ", qty="1", side="buy", type="stop",
                             created_at=now - timedelta(days=1))
    client = Client(orders=[old, recent], daytrade_count=2)
    report = correr_pre_apertura(session, AlpacaBroker(client), now)
    assert client.cancelled == ["old"]
    assert report["entradas_vencidas"] == ["old"]
    assert report["avisos"] and "PDT" in report["avisos"][0]
    client.daytrade_count = 1
    report = correr_pre_apertura(session, AlpacaBroker(client), now)
    assert report["avisos"] == []
    session.close(); engine.dispose()


def test_cli_exit_code_tracks_missing_stops(monkeypatch):
    import quantagent.cli.reconcile as reconcile
    from quantagent.cli.__main__ import cli

    class Components:
        broker = object()

    @contextmanager
    def scope():
        yield object()

    monkeypatch.setattr(reconcile.utils, "session_scope", scope)
    monkeypatch.setattr(reconcile.StrategyAssembler, "from_profiles", lambda **kwargs: object())
    monkeypatch.setattr(reconcile.StrategyAssembler, "build_components", lambda *a, **k: Components())
    monkeypatch.setattr(reconcile, "correr_pre_apertura", lambda *a: {"reenviados": [],
        "entradas_vencidas": [], "sin_stop": ["SPY"], "avisos": []})
    result = CliRunner().invoke(cli, ["trading", "pre-apertura"])
    assert result.exit_code == 1 and "ALARMA" in result.output
    monkeypatch.setattr(reconcile, "correr_pre_apertura", lambda *a: {"reenviados": [],
        "entradas_vencidas": [], "sin_stop": [], "avisos": []})
    result = CliRunner().invoke(cli, ["trading", "pre-apertura"])
    assert result.exit_code == 0
