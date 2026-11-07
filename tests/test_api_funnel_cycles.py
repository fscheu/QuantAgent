from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app
from quantagent.database import Base
from quantagent.models import Hypothesis, HypothesisAttempt, HypothesisEvent, Cycle


def test_cycle_history_review_endpoints_and_current_fallback(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'registry.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Cycle.__table__, Hypothesis.__table__,
        HypothesisEvent.__table__, HypothesisAttempt.__table__])
    with Session(engine) as db:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        db.add(Hypothesis(id="H", created_at=now))
        db.add_all([
            HypothesisEvent(ts=now, hypothesis_id=None, event="etapa", stage="1",
                verdict="CORTA", cycle=None, data={}),
            HypothesisEvent(ts=now, event="ciclo", cycle="2026-01",
                data={"accion": "abrir", "hipotesis": ["H"]}),
            HypothesisEvent(ts=now, hypothesis_id="H", event="etapa", stage="0",
                verdict="PASA", cycle="2026-01", data={}),
        ])
        db.commit()
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    client = TestClient(app)
    cycles = client.get("/api/funnel/cycles?limite=2")
    assert cycles.status_code == 200 and len(cycles.json()) == 1
    assert [m["key"] for m in cycles.json()[0]["milestones"]] == ["proposal", "funnel_start", "close"]
    assert client.get("/api/funnel/cycles/current").json()["cycle"] == "2026-01"
    assert client.get("/api/funnel/review").json()["limit"] == 6
    history = client.get("/api/funnel/history").json()
    assert set(history) == {"stage_events", "attempts", "events_by_stage", "attempts_by_label"}
    assert history["stage_events"] == 1 and history["events_by_stage"][1]["cut"] == 1
    assert client.get("/api/funnel/cycles?limite=0").status_code == 422
    engine.dispose()


def test_current_cycle_and_registry_errors(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'empty.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Hypothesis.__table__, HypothesisEvent.__table__,
        HypothesisAttempt.__table__, Cycle.__table__])
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    client = TestClient(app)
    assert client.get("/api/funnel/cycles/current").status_code == 404
    monkeypatch.delenv("QUANTAGENT_REGISTRO_URL")
    response = client.get("/api/funnel/history")
    assert response.status_code == 503 and "QUANTAGENT_REGISTRO_URL" in response.json()["detail"]
    engine.dispose()
