from datetime import datetime, timezone
from datetime import date

from click.testing import CliRunner
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from apps.api.app import account_dividends, feed as feed_route
from apps.api.cli import api_group
from quantagent.database import Base
from quantagent.models import Account, Event, Order, OrderSide, OrderStatus, OrderType, Trade


def test_sync_events_is_idempotent_and_visible_through_api(tmp_path, monkeypatch):
    account_path = tmp_path / "account.db"
    master_path = tmp_path / "master.db"
    account_engine = create_engine(f"sqlite:///{account_path}")
    master_engine = create_engine(f"sqlite:///{master_path}")
    Base.metadata.create_all(account_engine)
    Base.metadata.create_all(master_engine)
    with Session(account_engine) as source:
        source.add_all([
            Order(id=10, symbol="ABC", side=OrderSide.BUY, order_type=OrderType.MARKET,
                  quantity=5, filled_quantity=5, average_fill_price=12, status=OrderStatus.FILLED,
                  created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2),
                  filled_at=datetime(2026, 1, 2), comment=None),
            Order(id=11, symbol="XYZ", side=OrderSide.SELL, order_type=OrderType.STOP,
                  quantity=2, filled_quantity=2, average_fill_price=9, status=OrderStatus.FILLED,
                  created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2),
                  filled_at=datetime(2026, 1, 2), comment="stop triggered"),
            Order(id=12, symbol="BAD", side=OrderSide.BUY, order_type=OrderType.MARKET,
                  quantity=1, filled_quantity=0, status=OrderStatus.REJECTED,
                  created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2), comment="Buying power"),
            Trade(id=20, symbol="ABC", entry_price=10, exit_price=12, quantity=5,
                  side=OrderSide.BUY, pnl=10, pnl_pct=0.2, opened_at=datetime(2026, 1, 1),
                  closed_at=datetime(2026, 1, 2), notes=None),
        ])
        source.commit()
    with Session(master_engine) as master:
        master.add(Account(id="paper-a", name="Paper A", env="paper", kind="alpaca",
                           database_url=f"sqlite:///{account_path}", strategy="s", status="active"))
        master.commit()
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{master_path}")
    runner = CliRunner()
    assert runner.invoke(api_group, ["sync-events"]).exit_code == 0
    assert runner.invoke(api_group, ["sync-events"]).exit_code == 0
    with Session(master_engine) as master:
        events = master.scalars(select(Event).order_by(Event.type)).all()
        assert len(events) == 4
        assert {event.type for event in events} == {"order_filled", "stop_triggered", "order_rejected", "position_closed"}
        assert all(event.occurred_at.tzinfo is None or event.occurred_at.tzinfo == timezone.utc for event in events)
        assert next(e for e in events if e.type == "order_rejected").payload["reason"] == "Buying power"
        assert next(e for e in events if e.type == "order_filled").payload == {
            "symbol": "ABC", "side": "buy", "quantity": 5.0, "price": 12.0}
        assert next(e for e in events if e.type == "position_closed").payload["pnl"] == 10.0
    with Session(master_engine) as master:
        feed = feed_route(since=datetime(2026, 1, 1, tzinfo=timezone.utc), limit=50, db=master)
    assert {item.type for item in feed} >= {"order_filled", "stop_triggered", "order_rejected", "position_closed"}


def test_dividends_endpoint_reads_registered_event(tmp_path, monkeypatch):
    path = tmp_path / "master.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine, tables=[Account.__table__, Event.__table__])
    with Session(engine) as db:
        db.add(Account(id="paper-a", name="Paper A", env="paper", kind="alpaca", database_url="sqlite:///x", strategy="s"))
        from quantagent.trading.alpaca_dividendos import registrar_dividendos_evento
        dividends = [("ABC", "2.5", "USD", "efectivo en cuenta")]
        assert registrar_dividendos_evento(db, "paper-a", date(2026, 1, 2), dividends)
        assert not registrar_dividendos_evento(db, "paper-a", date(2026, 1, 2), dividends)
        db.commit()
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{path}")
    with Session(engine) as db:
        response = account_dividends("paper-a", db)
    assert response[0].symbols == ["ABC"]
    assert response[0].net == 2.5
