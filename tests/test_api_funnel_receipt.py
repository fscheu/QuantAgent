from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app
from quantagent.database import Base
from quantagent.models import Cycle, Hypothesis, HypothesisAttempt, HypothesisEvent


def test_receipt_endpoint_open_closed_and_validation(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'registry.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[Cycle.__table__, Hypothesis.__table__, HypothesisEvent.__table__, HypothesisAttempt.__table__])
    with Session(engine) as db:
        from datetime import datetime, timezone
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        db.add(Hypothesis(id="H", created_at=now))
        db.add(Cycle(cycle="2026-01", opened_at=now, hypothesis_ids=["H"], receipt=None))
        db.add_all([HypothesisEvent(ts=now, event="ciclo", cycle="2026-01", data={"accion": "abrir", "hipotesis": ["H"]}),
                    HypothesisEvent(ts=now, event="ciclo", cycle="2026-01", data={"accion": "cerrar"}),
                HypothesisEvent(ts=now, hypothesis_id="H", event="etapa", stage="0", verdict="CORTA", cycle="2026-01", reason="motivo", data={"measure": {"metric": "sharpe", "value": 0.2, "threshold": 0.5, "unit": "ratio", "comparator": ">="}})])
        db.commit()
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    client = TestClient(app)
    partial = client.get("/api/funnel/cycles/2026-01/receipt").json()
    assert partial["partial"] and partial["id"] is None and partial["hash"] is None
    assert partial["drops"][0]["measure"]["value"] == 0.2
    assert set(partial) == {"id", "hash", "cycle", "partial", "generated_at", "counts", "stages", "drops", "attempts", "snapshots"}
    with Session(engine) as db:
        row = db.get(Cycle, "2026-01")
        row.closed_at = now
        row.receipt = {"version": 1, "ciclo": "2026-01", "hipotesis": ["H"], "etapas": {"0": 1}, "promovidas": 0, "caidas": {"H": {"etapa": "0", "veredicto": "CORTA", "motivo": "motivo"}}}
        db.commit()
    legacy = client.get("/api/funnel/cycles/2026-01/receipt")
    assert legacy.status_code == 200 and legacy.json()["id"] == "2026-01" and legacy.json()["hash"] is None
    assert client.get("/api/funnel/cycles/nope/receipt").status_code == 422
    assert client.get("/api/funnel/cycles/2026-02/receipt").status_code == 404
    engine.dispose()
