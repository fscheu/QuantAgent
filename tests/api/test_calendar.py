from datetime import date, time

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import calendar as calendar_route
from apps.api.calendar_sync import sync_calendar
from quantagent.database import Base
from quantagent.models import CalendarEvent


class AlpacaDay:
    def __init__(self, day, close):
        self.date = date.fromisoformat(day)
        self.close = time.fromisoformat(close)


def test_sync_upserts_holiday_and_early_close_idempotently(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'calendar.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[CalendarEvent.__table__])
    days = [AlpacaDay("2026-11-25", "16:00"), AlpacaDay("2026-11-27", "13:00"),
            AlpacaDay("2026-11-30", "16:00")]
    with Session(engine) as db:
        assert sync_calendar(db, lambda: days) == 2
        assert sync_calendar(db, lambda: days) == 2
        events = db.query(CalendarEvent).order_by(CalendarEvent.date, CalendarEvent.kind).all()
        assert [(e.date.isoformat(), e.kind, e.source) for e in events] == [
            ("2026-11-26", "holiday", "alpaca"), ("2026-11-27", "early_close", "alpaca")]


def test_calendar_endpoint_filters_window_and_formats_dates(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'calendar.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[CalendarEvent.__table__])
    with Session(engine) as db:
        db.add_all([
            CalendarEvent(date=date(2026, 11, 26), kind="holiday", title="Thanksgiving", source="alpaca"),
            CalendarEvent(date=date(2026, 12, 1), kind="early_close", title="Early close", source="alpaca"),
            CalendarEvent(date=date(2027, 1, 15), kind="holiday", title="Future holiday", source="alpaca"),
        ])
        db.commit()

    with Session(engine) as db:
        result = calendar_route(days=30, db=db)
    assert result[0].date == "2026-11-26"
    assert result[0].kind == "holiday"
    assert len(result) == 1
