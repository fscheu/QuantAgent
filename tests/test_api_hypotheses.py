from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app
from quantagent.database import Base
from quantagent.models import Hypothesis, HypothesisAttempt, HypothesisEvent


def test_hypotheses_endpoint_filters_and_contract(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'registry.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Hypothesis.__table__, HypothesisEvent.__table__, HypothesisAttempt.__table__])
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        for name, stage, verdict in (("active", "0", "PASA"), ("promoted", "3", "PASA"),
                                     ("discarded", "1", "CORTA"), ("held", "2", "ERROR")):
            db.add(Hypothesis(id=name, created_at=now))
            db.add(HypothesisEvent(ts=now, hypothesis_id=name, event="etapa", stage=stage,
                                   verdict=verdict, reason="test", cycle="2026-11", data={}))
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        db.add(Hypothesis(id="old", created_at=old))
        db.add(HypothesisEvent(ts=old, hypothesis_id="old", event="etapa", stage="0",
                               verdict="PASA", reason="test", cycle=None, data={}))
        db.add(HypothesisEvent(ts=now, event="ciclo", cycle="2026-11",
                               data={"accion": "abrir", "hipotesis": ["active", "promoted", "discarded", "held"]}))
        db.commit()
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    client = TestClient(app)
    rows = client.get("/api/hypotheses").json()
    assert len(rows) == 5
    assert set(rows[0]) == {"id", "name", "idea", "cycle", "stage", "verdict", "status", "promoted",
                            "used_reserve", "attempts_valid", "attempts_invalid", "last_metric",
                            "discard_reason", "discard_measure", "updated_at"}
    assert {row["status"] for row in rows} == {"active", "promoted", "discarded", "held"}
    # usó la reserva = llegó a la etapa 3 (holdout)
    assert {row["id"]: row["used_reserve"] for row in rows if row["id"] != "old"}["promoted"] is True
    assert {row["id"]: row["used_reserve"] for row in rows}["active"] is False
    assert len(client.get("/api/hypotheses?estado=held&etapa=e2&ciclo=2026-11&periodo=30d").json()) == 1
    recent = client.get("/api/hypotheses?periodo=30d").json()
    assert len(recent) == 4 and all(row["id"] != "old" for row in recent)
    assert client.get("/api/hypotheses?periodo=7d").status_code == 422
    assert client.get("/api/hypotheses?etapa=e9").status_code == 422
    monkeypatch.delenv("QUANTAGENT_REGISTRO_URL")
    missing = client.get("/api/hypotheses")
    assert missing.status_code == 503 and "QUANTAGENT_REGISTRO_URL" in missing.json()["detail"]
    engine.dispose()
