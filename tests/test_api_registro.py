import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text

from apps.api.registro import registro_session
from quantagent.database import Base
from quantagent.models import Cycle, Hypothesis, HypothesisAttempt, HypothesisEvent


def _client():
    app = FastAPI()

    @app.get("/registro")
    def read_registro(db=Depends(registro_session)):
        return {"count": len(db.scalars(select(Hypothesis)).all())}

    return TestClient(app)


def test_registro_reads_and_rejects_writes(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'registro.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine, tables=[model.__table__ for model in (Hypothesis, HypothesisEvent, HypothesisAttempt, Cycle)])
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO hypotheses (id, created_at) VALUES ('h1', CURRENT_TIMESTAMP)"))
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", url)
    client = _client()
    assert client.get("/registro").json() == {"count": 1}
    from apps.api.registro import _session_factory

    db = _session_factory()()
    try:
        with pytest.raises(Exception, match="readonly"):
            db.execute(text("INSERT INTO hypotheses (id, created_at) VALUES ('h2', CURRENT_TIMESTAMP)"))
    finally:
        db.close()
        engine.dispose()


def test_registro_missing_url_returns_503(monkeypatch):
    monkeypatch.delenv("QUANTAGENT_REGISTRO_URL", raising=False)
    response = _client().get("/registro")
    assert response.status_code == 503
    assert "QUANTAGENT_REGISTRO_URL" in response.json()["detail"]


def test_registro_unreachable_database_returns_503(monkeypatch):
    monkeypatch.setenv("QUANTAGENT_REGISTRO_URL", "postgresql://invalid:invalid@127.0.0.1:1/registro")
    response = _client().get("/registro")
    assert response.status_code == 503
    assert "base del registro" in response.json()["detail"]
