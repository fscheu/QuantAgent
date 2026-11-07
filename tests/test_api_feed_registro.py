from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app
from quantagent.database import Base
from quantagent.models import Account, Cycle, Event


def _engine(path):
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[Account.__table__, Event.__table__, Cycle.__table__])
    return engine


def _cycle(cycle, closed, receipt, opened=None):
    return Cycle(cycle=cycle, opened_at=opened or closed - timedelta(days=30), closed_at=closed,
                 hypothesis_ids=receipt.get("hipotesis", []), receipt=receipt)


def test_feed_derives_cycle_events_and_review_streak(tmp_path, monkeypatch):
    master, registry = _engine(tmp_path / "master.db"), _engine(tmp_path / "registry.db")
    now = datetime(2026, 11, 30, 15, tzinfo=timezone.utc)
    with Session(registry) as db:
        db.add_all([_cycle("2026-05", now - timedelta(days=180), {"hipotesis": ["a"], "promovidas": 1}),
                    *[_cycle(f"2026-{month:02}", now - timedelta(days=30 * (11-month)),
                             {"version": 2, "hipotesis": ["a", "b"], "promovidas": 0,
                              "por_hipotesis": {"a": {"estado_final": "cortada"},
                                                "b": {"estado_final": "detenida"}}})
                      for month in range(6, 12)],
                    Cycle(cycle="2026-12", opened_at=now, closed_at=None, hypothesis_ids=[], receipt=None)])
        db.commit()
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'master.db'}")
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", f"sqlite:///{tmp_path / 'registry.db'}")
    rows = TestClient(app).get("/api/feed", params={"since": (now - timedelta(days=10)).isoformat()}).json()
    closed = next(row for row in rows if row["id"] == "cycle_closed:2026-11")
    assert (closed["type"], closed["severity"], closed["account_id"], closed["group_count"]) == ("cycle_closed", "info", None, 1)
    assert closed["subtitle"] == "2 entraron · 2 cayeron · 0 promovidas"
    assert not any(row["id"] == "cycle_closed:2026-10" for row in rows)
    rule = next(row for row in rows if row["id"] == "review_rule_reached:2026-11")
    assert rule["title"] == "Embudo: 6 ciclos seguidos sin promovida"
    assert not any("2026-12" in row["id"] for row in rows)
    home = TestClient(app).get("/api/home/status")
    assert home.status_code == 200 and home.json()["level"] == "ok"
    with Session(master) as db:
        assert db.query(Event).count() == 0


def test_feed_registry_optional_and_v1_receipts(tmp_path, monkeypatch):
    master, registry = _engine(tmp_path / "master.db"), _engine(tmp_path / "registry.db")
    now = datetime.now(timezone.utc)
    with Session(registry) as db:
        db.add_all([_cycle("2026-10", now - timedelta(minutes=10),
                           {"hipotesis": ["a", "b"], "promovidas": 0,
                            "caidas": {"a": {"veredicto": "CORTA"}, "b": {"veredicto": "ERROR"}}}),
                    _cycle("2026-11", now, {"hipotesis": ["c"], "promovidas": 1})])
        db.commit()
    monkeypatch.setenv("QUANTAGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'master.db'}")
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", f"sqlite:///{tmp_path / 'registry.db'}")
    rows = TestClient(app).get("/api/feed").json()
    v1 = next(row for row in rows if row["id"] == "cycle_closed:2026-10")
    promoted = next(row for row in rows if row["id"] == "cycle_closed:2026-11")
    assert v1["subtitle"] == "2 entraron · 2 cayeron · 0 promovidas"
    assert promoted["subtitle"] == "1 entraron · 0 cayeron · 1 promovidas"
    monkeypatch.delenv("QUANTAGENT_REGISTRO_URL")
    assert TestClient(app).get("/api/feed").status_code == 200
