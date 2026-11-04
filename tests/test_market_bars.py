from datetime import date, datetime

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quantagent.data.market_bars import backfill_snapshot, market_bar_status, present_sessions, upsert_bars
from quantagent.data.snapshot import write_snapshot
from quantagent.models import Base, MarketBar


def test_market_bar_status_uses_last_closed_nyse_session_and_holiday_calendar():
    # July 4 is closed; before the July 5 close, July 3 remains the latest session.
    session, stale, missing = market_bar_status({date(2024, 7, 3)}, datetime(2024, 7, 5, 14))
    assert (session, stale, missing) == (date(2024, 7, 3), False, [])


def test_market_bar_status_skips_weekend_and_marks_after_close_stale():
    session, stale, missing = market_bar_status({date(2024, 7, 3)}, datetime(2024, 7, 8, 22))
    assert session == date(2024, 7, 8)
    assert stale
    assert missing == [date(2024, 7, 5), date(2024, 7, 8)]


def test_market_bar_status_reports_intermediate_gaps():
    session, stale, missing = market_bar_status(
        {date(2024, 1, 2), date(2024, 1, 4)}, datetime(2024, 1, 5, 22)
    )
    assert session == date(2024, 1, 5)
    assert stale
    assert missing == [date(2024, 1, 3), date(2024, 1, 5)]


def test_upsert_bars_is_idempotent_and_updates_changed_values():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    rows = pd.DataFrame([{"open": 1, "high": 2, "low": 1, "close": 2, "volume": 10}], index=pd.to_datetime(["2024-01-02"]))
    with Session(engine) as session:
        upsert_bars(session, "SPY", rows, "fixture")
        original = session.get(MarketBar, ("SPY", date(2024, 1, 2)))
        upsert_bars(session, "SPY", rows, "fixture")
        assert len(session.scalars(select(MarketBar)).all()) == 1
        assert session.get(MarketBar, ("SPY", date(2024, 1, 2))).close == original.close
        rows.loc[pd.Timestamp("2024-01-02"), "close"] = 3
        upsert_bars(session, "SPY", rows, "fixture")
        assert session.get(MarketBar, ("SPY", date(2024, 1, 2))).close == 3


def test_backfill_snapshot_loads_rows_and_reports_missing_symbols(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    frame = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [2], "volume": [10], "adj_close": [2]}, index=pd.to_datetime(["2024-01-02"]))
    write_snapshot("bars", {"SPY": frame}, base_dir=tmp_path)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    with Session(engine) as session:
        missing = backfill_snapshot(session, "bars")
        assert missing and "SPY" not in missing
        assert present_sessions(session, "SPY") == {date(2024, 1, 2)}


def test_present_sessions_feeds_market_bar_status_from_sqlite():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    with Session(engine) as session:
        upsert_bars(session, "SPY", pd.DataFrame([{"open": 1, "high": 2, "low": 1, "close": 2, "volume": 1}], index=pd.to_datetime(["2024-07-03"])), "fixture")
        assert market_bar_status(present_sessions(session, "SPY"), datetime(2024, 7, 5, 14)) == (date(2024, 7, 3), False, [])
