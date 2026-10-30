import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from apps.api.app import app, session_factory


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_api_session_rejects_writes(tmp_path):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE sample (id INTEGER)"))
    db = session_factory(url)()
    try:
        with pytest.raises(Exception, match="readonly"):
            db.execute(text("INSERT INTO sample (id) VALUES (1)"))
    finally:
        db.close()
        engine.dispose()


def test_serve_defaults_to_loopback(monkeypatch):
    from click.testing import CliRunner
    import apps.api.cli as api_cli

    seen = {}
    monkeypatch.setattr(api_cli.uvicorn, "run", lambda *a, **kw: seen.update(kw))
    result = CliRunner().invoke(api_cli.serve, [])
    assert result.exit_code == 0
    assert seen["host"] == "127.0.0.1"
