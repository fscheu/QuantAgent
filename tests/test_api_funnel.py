from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app
from quantagent.database import Base
from quantagent.models import Hypothesis, HypothesisEvent


def test_funnel_reads_readonly_registry_and_filters_cycles(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'registry.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Hypothesis.__table__, HypothesisEvent.__table__])
    with Session(engine) as db:
        for name, verdict in zip("ABCD", ("PASA", "CORTA", "ERROR", "ALARMA")):
            db.add(Hypothesis(id=name, created_at=datetime.now(timezone.utc)))
            db.add(HypothesisEvent(ts=datetime.now(timezone.utc), hypothesis_id=name, event="etapa",
                                   stage="0", verdict=verdict, reason="test", cycle="2026-01",
                                   data={"cause_code": "test"}))
        db.add(HypothesisEvent(ts=datetime.now(timezone.utc), hypothesis_id="importada", event="etapa",
                               stage="0", verdict="PASA", reason="", cycle=None, data={}))
        db.add(HypothesisEvent(ts=datetime.now(timezone.utc), event="ciclo", cycle="2026-01",
                               data={"accion": "abrir", "hipotesis": list("ABCD")}))
        db.commit()
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    client = TestClient(app)

    all_cycles = client.get("/api/funnel?ciclo=all&periodo=all").json()
    assert set(all_cycles) == {"cycle", "period", "total", "stages", "later", "promoted", "drop_reasons", "sample_too_small"}
    assert all_cycles["total"] == 4
    assert {reason["verdict"] for reason in all_cycles["drop_reasons"]} == {"CORTA", "ERROR", "ALARMA"}
    assert client.get("/api/funnel?ciclo=2026-01&periodo=30d").json()["total"] == 4
    engine.dispose()


def test_funnel_invalid_period_and_missing_registry_return_errors(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", "sqlite:///:memory:")
    response = TestClient(app).get("/api/funnel?periodo=7d")
    assert response.status_code == 422 and "periodo" in response.json()["detail"]
    monkeypatch.delenv("QUANTAGENT_REGISTRO_URL")
    response = TestClient(app).get("/api/funnel")
    assert response.status_code == 503 and "QUANTAGENT_REGISTRO_URL" in response.json()["detail"]
