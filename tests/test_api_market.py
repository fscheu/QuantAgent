import re
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app, get_db
from quantagent.database import Base
from quantagent.models import Account, CalendarEvent, MarketBar


def test_market_page_contract_and_independent_sections(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'maestro.db'}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__, Account.__table__, CalendarEvent.__table__])
    with Session(engine) as db:
        db.add(Account(id="paper-a", name="Paper A", env="paper", kind="simulated", database_url=f"sqlite:///{tmp_path / 'maestro.db'}", strategy="test", status="active"))
        db.commit()
        app.dependency_overrides[get_db] = lambda: db
        monkeypatch.setattr("apps.api.market_regime.regimen_spy", lambda _: (_ for _ in ()).throw(ValueError("broken")))
        monkeypatch.setattr("apps.api.positions.read_positions", lambda *_: [SimpleNamespace(symbol="SPY", status="executed")])
        monkeypatch.setattr("apps.api.app.calendar", lambda days, db: [])
        try:
            payload = TestClient(app).get("/api/market").json()
        finally:
            app.dependency_overrides.clear()
    expected = {"as_of", "clock", "regime", "motor", "sectors", "macro", "recessions", "etfs", "events", "errors"}
    assert set(payload) == expected
    assert payload["regime"] is None and payload["errors"]["regime"] == "broken"
    assert isinstance(payload["etfs"], list) and payload["errors"] == {"regime": "broken"}
    assert payload["etfs"][0]["held_in"] == [{"account_id": "paper-a", "account_name": "Paper A"}]
    config = __import__("pathlib").Path("docs/03_design/web-design-system/config.ts").read_text()
    macro_config = config.split("macroSeries:", 1)[1].split("],", 1)[0]
    configured_codes = re.findall(r"[A-Z][A-Z0-9]{2,}", macro_config)
    assert [series["code"] for series in payload["macro"]] == configured_codes
    types = (tmp_path.parents[2] / "docs/03_design/web-design-system/api-types.ts")
    if not types.exists():
        types = __import__("pathlib").Path("docs/03_design/web-design-system/api-types.ts")
    content = types.read_text()
    declared = set(re.findall(r"^  (\w+)\??:", content.split("export interface MarketPage {")[1].split("\n}")[0], re.M))
    assert expected == declared
