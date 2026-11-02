from datetime import date, datetime as RealDateTime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api import app as api_module
from apps.api.app import calendar as calendar_route
from apps.api.calendar_fred import sync_fred_releases, sync_manual_events
from quantagent.database import Base
from quantagent.models import Account, CalendarEvent


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'calendar.db'}")
    Base.metadata.create_all(engine, tables=[CalendarEvent.__table__, Account.__table__])
    with Session(engine) as session:
        yield session


def test_fred_releases_upsert_idempotently_and_continue_after_failure(db_session):
    dates = {50: "2026-11-06", 54: "2026-11-25"}

    def fetch(release_id):
        if release_id == 10:
            raise RuntimeError("temporary FRED failure")
        return [{"release_id": release_id, "date": dates[release_id]}]

    assert sync_fred_releases(db_session, fetch) == 2
    assert sync_fred_releases(db_session, fetch) == 2
    rows = db_session.query(CalendarEvent).filter_by(source="fred").order_by(CalendarEvent.date).all()
    assert [(row.date.isoformat(), row.kind, row.title) for row in rows] == [
        ("2026-11-06", "macro", "Empleo (nonfarm payrolls)"),
        ("2026-11-25", "macro", "PCE"),
    ]


def test_manual_yaml_validates_and_removes_deleted_entries(db_session, tmp_path):
    path = tmp_path / "manual.yaml"
    path.write_text('events:\n  - date: "2026-11-10"\n    title: "FOMC"\n')
    assert sync_manual_events(db_session, path) == 1
    assert sync_manual_events(db_session, path) == 1
    path.write_text("events: []\n")
    assert sync_manual_events(db_session, path) == 0
    assert db_session.query(CalendarEvent).filter_by(source="manual").count() == 0

    path.write_text('events:\n  - date: "2026-2-3"\n    title: "FOMC"\n')
    with pytest.raises(ValueError, match="evento manual inválido"):
        sync_manual_events(db_session, path)
    path.write_text('events:\n  - date: "2026-02-03"\n    title: " "\n')
    with pytest.raises(ValueError, match="evento manual inválido"):
        sync_manual_events(db_session, path)


def test_calendar_returns_macro_events_only_inside_requested_window(db_session, monkeypatch):
    class FixedDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(api_module, "datetime", FixedDateTime)
    db_session.add_all([
        CalendarEvent(date=date(2026, 11, 2), kind="macro", title="IPC", source="fred"),
        CalendarEvent(date=date(2026, 11, 30), kind="macro", title="FOMC", source="manual"),
        CalendarEvent(date=date(2026, 12, 1), kind="macro", title="Fuera", source="manual"),
    ])
    db_session.commit()

    result = calendar_route(days=30, db=db_session)
    assert [(row.date, row.kind, row.source) for row in result] == [
        ("2026-11-02", "macro", "fred"),
        ("2026-11-30", "macro", "manual"),
    ]
