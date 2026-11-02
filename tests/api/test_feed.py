from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app, feed as feed_route
from quantagent.database import Base
from quantagent.models import Account, Event


def _db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'feed.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[Account.__table__, Event.__table__])
    return engine


def _event(kind, at, account="a", entity=None, severity="warning", payload=None):
    return Event(type=kind, severity=severity, occurred_at=at, account_id=account,
                 entity_id=entity, payload=payload or {})


def test_feed_groups_by_day_type_and_account_and_orders_severity_recency(tmp_path):
    engine = _db(tmp_path)
    now = datetime(2026, 10, 31, 15, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(Account(id="a", name="Cuenta A", env="paper", kind="simulated", database_url="sqlite://",
                       strategy="s"))
        db.add_all([
            _event("order_rejected", now - timedelta(hours=1), payload={"reason": "Poder"}),
            _event("order_rejected", now - timedelta(minutes=30), payload={"reason": "Margen"}),
            _event("order_rejected", now - timedelta(days=1), payload={"reason": "Otra"}),
            _event("drawdown_limit", now - timedelta(days=2), severity="critical"),
            _event("stop_near", now - timedelta(minutes=2), entity="pos-1", severity="warning"),
            _event("stop_near", now - timedelta(minutes=1), entity="pos-2", severity="warning"),
        ])
        db.commit()
    with Session(engine) as db:
        rows = feed_route(since=now - timedelta(days=5), limit=50, db=db)
    assert rows[0].severity == "critical"
    rejected = [r for r in rows if r.type == "order_rejected"]
    assert sorted(r.group_count for r in rejected) == [1, 2]
    assert rows.index(next(r for r in rows if r.type == "stop_near" and r.id == "6")) < rows.index(next(r for r in rows if r.type == "stop_near" and r.id == "5"))


def test_feed_since_limit_and_incomplete_payload(tmp_path):
    engine = _db(tmp_path)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        db.add_all([_event("order_rejected", now - timedelta(days=2)),
                    _event("stop_breached", now, payload={})])
        db.commit()
    with Session(engine) as db:
        rows = feed_route(since=now - timedelta(hours=1), limit=1, db=db)
    assert len(rows) == 1 and rows[0].type == "stop_breached" and rows[0].subtitle == "cierre pendiente"


def test_feed_templates_use_all_events_in_group(tmp_path):
    engine = _db(tmp_path)
    now = datetime(2026, 10, 31, 15, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(Account(id="a", name="Cuenta A", env="paper", kind="simulated", database_url="sqlite://",
                       strategy="s"))
        db.add_all([
            _event("order_rejected", now - timedelta(minutes=2), payload={"reason": "Poder"}),
            _event("order_rejected", now - timedelta(minutes=1), payload={"reason": "Margen"}),
            _event("stop_triggered", now - timedelta(minutes=4), payload={"symbol": "MSFT"}),
            _event("stop_triggered", now - timedelta(minutes=3), payload={"symbol": "SPY"}),
            _event("order_filled", now - timedelta(minutes=2), payload={"symbol": "MSFT", "side": "buy", "quantity": 25, "price": 431.55}),
            _event("order_filled", now - timedelta(minutes=1), payload={"symbol": "SPY", "side": "buy", "quantity": 4, "price": 565.98}),
        ])
        db.commit()
    with Session(engine) as db:
        rows = feed_route(since=now - timedelta(hours=1), limit=50, db=db)
    rejected = next(row for row in rows if row.type == "order_rejected")
    stops = next(row for row in rows if row.type == "stop_triggered")
    filled = next(row for row in rows if row.type == "order_filled")
    assert (rejected.title, rejected.subtitle) == ("2 órdenes rechazadas en Cuenta A", "Poder, Margen")
    assert stops.title == "Saltaron 2 stops en Cuenta A"
    assert (filled.title, filled.subtitle) == ("2 compras ejecutadas", "MSFT 25 a 431,55 · SPY 4 a 565,98")


def test_feed_deduplicates_active_alarm_but_shows_resolved_pair(tmp_path):
    engine = _db(tmp_path)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        db.add_all([
            _event("stop_near", now - timedelta(minutes=4), entity="p1"),
            _event("stop_near", now - timedelta(minutes=3), entity="p1"),
            _event("stop_near", now - timedelta(minutes=2), entity="p2"),
            _event("stop_near", now - timedelta(minutes=1), entity="p2"),
            _event("alarm_resolved", now, entity="p2", severity="neutral", payload={"alarm_type": "stop_near"}),
        ])
        db.commit()
    with Session(engine) as db:
        rows = feed_route(since=now - timedelta(hours=1), limit=50, db=db)
    assert len(rows) == 3
    assert sorted(r.type for r in rows) == ["alarm_resolved", "stop_near", "stop_near"]
    assert next(r for r in rows if r.type == "stop_near" and r.id == "1").group_count == 1


def test_feed_endpoint_session_is_read_only(tmp_path, monkeypatch):
    engine = _db(tmp_path)
    with Session(engine) as db:
        db.add(_event("order_rejected", datetime.now(timezone.utc)))
        db.commit()
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'feed.db'}")
    client = TestClient(app)
    assert client.get("/api/feed").status_code == 200
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM events").scalar_one() == 1
