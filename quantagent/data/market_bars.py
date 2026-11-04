"""Pure status calculation for the daily market bar store."""

from datetime import date, datetime, timedelta, timezone

import pandas_market_calendars as mcal


def market_bar_status(present_sessions: set[date], now: datetime) -> tuple[date, bool, list[date]]:
    """Return latest closed NYSE session, whether it is stale, and intervening gaps."""
    calendar = mcal.get_calendar("NYSE")
    today = now.date()
    schedule = calendar.schedule(start_date=today - timedelta(days=14), end_date=today)
    current = now
    if current.tzinfo is None:
        current = current.replace(tzinfo=calendar.tz)
    current = current.astimezone(timezone.utc)
    closed = [day.date() for day, row in schedule.iterrows() if row["market_close"] <= current]
    session_date = closed[-1]
    start = min(present_sessions) if present_sessions else session_date
    expected = calendar.valid_days(start_date=start, end_date=session_date)
    missing = [day.date() for day in expected if day.date() not in present_sessions]
    return session_date, session_date not in present_sessions, missing
