"""Synchronize Alpaca market holidays and early closes."""

import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.models import CalendarEvent

NEW_YORK = ZoneInfo("America/New_York")


def sync_calendar(session: Session, fetcher):
    """Upsert holidays and early closes from a callable returning Alpaca calendar days."""
    rows = sorted(fetcher(), key=lambda row: _day(row))
    events = []
    for previous, current in zip(rows, rows[1:]):
        day = _day(previous) + timedelta(days=1)
        while day < _day(current):
            if day.weekday() < 5:
                events.append((day, "holiday", "Market holiday"))
            day += timedelta(days=1)
    for row in rows:
        close = _local_time(row.close)
        if close < time(16, 0):
            events.append((_day(row), "early_close", "Early close"))

    for day, kind, title in events:
        event = session.query(CalendarEvent).filter_by(date=day, kind=kind, source="alpaca").one_or_none()
        if event is None:
            session.add(CalendarEvent(date=day, kind=kind, title=title, source="alpaca"))
        else:
            event.title = title
    session.commit()
    return len(events)


def _day(row):
    value = row.date
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _local_time(value):
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.astimezone(NEW_YORK).time()
    return time.fromisoformat(str(value))


def alpaca_calendar_fetcher():
    from alpaca.trading.client import TradingClient

    client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    today = datetime.now(NEW_YORK).date()
    return lambda: client.get_calendar(start=today.isoformat(), end=(today + timedelta(days=365)).isoformat())


def sync_from_alpaca():
    engine = create_engine(os.environ["QUANTAGENT_DATABASE_URL"])
    with Session(engine) as session:
        return sync_calendar(session, alpaca_calendar_fetcher())
