import json
from datetime import date, datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.accounts import add_account
from apps.api.app import account_dividends, account_equity, account_sector_exposure
from quantagent.data.snapshot import write_snapshot
from quantagent.database import Base
from quantagent.models import Account, EquitySnapshot, Event, OrderSide, Position


def registry(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'master.db'}")
    Base.metadata.create_all(engine, tables=[Account.__table__, Event.__table__])
    return engine


def add_registry_account(session, account_id, database_url):
    return add_account(session, id=account_id, name=account_id, env="paper", kind="simulated",
        database_url=database_url, strategy="test")


def test_equity_reads_scaled_spy_and_account_shadow(tmp_path, monkeypatch):
    today = date.today()
    account_path = tmp_path / "paper.db"
    account_url = f"sqlite:///{account_path}"
    engine = create_engine(account_url)
    EquitySnapshot.__table__.create(engine)
    with Session(engine) as source:
        source.add_all([EquitySnapshot(snapshot_date=today, taken_at=datetime.now(), equity=200, cash=0, invested=200),
                        EquitySnapshot(snapshot_date=date.fromordinal(today.toordinal()-1), taken_at=datetime.now(), equity=180, cash=0, invested=180)])
        source.commit()
    # Ensure chronological first row maps to the first SPY close.
    prices = pd.DataFrame({"open": [10, 12], "high": [10, 12], "low": [10, 12], "close": [10, 12],
        "adj_close": [10, 12], "volume": [1, 1]}, index=pd.to_datetime([date.fromordinal(today.toordinal()-1), today]))
    write_snapshot("web", {"SPY": prices}, base_dir=tmp_path / "snapshots")
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("QUANTAGENT_WEB_SNAPSHOT", "web")
    shadow = {today.isoformat(): {"paper": {"pesos": {}, "valor": 211}}}
    (tmp_path / "paper.sombra.json").write_text(json.dumps(shadow))
    master = registry(tmp_path)
    with Session(master) as db:
        add_registry_account(db, "acct", account_url)
        points = account_equity("acct", "ALL", db)
    assert [p.benchmark for p in points] == [180.0, 216.0]
    assert [p.shadow for p in points] == [None, 211.0]
    engine.dispose(); master.dispose()


def test_dividends_are_filtered_by_account(tmp_path):
    (tmp_path / 'paper.db').touch()
    engine = registry(tmp_path)
    with Session(engine) as db:
        add_registry_account(db, "acct", f"sqlite:///{tmp_path / 'paper.db'}")
        db.add_all([Event(type="dividend_received", severity="info", occurred_at=datetime(2026, 1, 1), account_id="acct",
                          payload={"symbols": ["SPY"], "net": 2.5, "withheld": .5}),
                    Event(type="dividend_received", severity="info", occurred_at=datetime(2026, 1, 2), account_id="other",
                          payload={"symbols": ["QQQ"], "net": 9, "withheld": 0})])
        db.commit()
        result = account_dividends("acct", db)
    assert len(result) == 1 and result[0].symbols == ["SPY"] and result[0].net == 2.5
    engine.dispose()


def test_sector_exposure_keeps_short_and_unclassified_etf_cash(tmp_path, monkeypatch):
    monkeypatch.delenv("QUANTAGENT_WEB_SNAPSHOT", raising=False)
    path = tmp_path / "paper.db"
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Position.__table__.create(engine); EquitySnapshot.__table__.create(engine)
    with Session(engine) as source:
        source.add_all([Position(symbol="ETF", quantity=2, average_entry_price=5, current_price=10, unrealized_pnl=0,
            unrealized_pnl_pct=0, side=OrderSide.BUY), Position(symbol="SHORT", quantity=-1, average_entry_price=20,
            current_price=20, unrealized_pnl=0, unrealized_pnl_pct=0, side=OrderSide.SELL)])
        source.add(EquitySnapshot(snapshot_date=date.today(), taken_at=datetime.now(), equity=40, cash=10, invested=30))
        source.commit()
    master = registry(tmp_path)
    with Session(master) as db:
        account = add_registry_account(db, "acct", url)
        from apps.api.accounts import account_session_factory
        with account_session_factory(account)() as source:
            result = account_sector_exposure("acct", db)
    by_sector = {row["sector"]: row for row in result}
    assert by_sector["Sin clasificar"]["long"] == 20
    assert by_sector["Sin clasificar"]["short"] == 20
    assert by_sector["Cash"]["long"] == 10
    engine.dispose(); master.dispose()
