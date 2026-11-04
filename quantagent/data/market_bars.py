"""Pure status calculation for the daily market bar store."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas_market_calendars as mcal
from sqlalchemy import select

from quantagent.data.snapshot import load_manifest, read_snapshot
from quantagent.models import MarketBar


MARKET_BAR_SYMBOLS = (
    "SPY", "QQQ", "IWM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "EFA", "EEM", "EWJ", "EWZ", "FXI", "SHY", "IEF", "TLT", "TIP", "AGG", "LQD",
    "GLD", "SLV", "DBC", "USO", "FXE", "VNQ", "XLC", "XLRE",
)


def upsert_bars(session, symbol: str, frame, source: str) -> None:
    """Insert or update daily OHLCV rows by symbol and session date."""
    for index, row in frame.iterrows():
        session_date = index.date() if hasattr(index, "date") else index
        values = {key: Decimal(str(row[key])) for key in ("open", "high", "low", "close", "volume")}
        bar = session.get(MarketBar, (symbol, session_date))
        if bar is None:
            session.add(MarketBar(symbol=symbol, session_date=session_date, source=source, **values))
        elif any(getattr(bar, key) != value for key, value in values.items()) or bar.source != source:
            for key, value in values.items():
                setattr(bar, key, value)
            bar.source = source
    session.flush()


def present_sessions(session, symbol: str) -> set[date]:
    """Return the stored daily sessions for one symbol."""
    return set(session.scalars(select(MarketBar.session_date).where(MarketBar.symbol == symbol)))


def backfill_snapshot(session, snapshot: str, source: str = "snapshot") -> list[str]:
    """Load available universe symbols from a frozen snapshot and return absent symbols."""
    manifest = load_manifest(snapshot)
    missing = [symbol for symbol in MARKET_BAR_SYMBOLS if symbol not in manifest["symbols"]]
    for symbol in MARKET_BAR_SYMBOLS:
        if symbol in manifest["symbols"]:
            upsert_bars(session, symbol, read_snapshot(snapshot, symbol), source)
    return missing


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


class DailyAlpacaClient:
    """Fetch daily bars as timestamp-indexed frames accepted by upsert_bars."""

    def __init__(self, source=None, base_url=None):
        from quantagent.data.sources.alpaca import AlpacaSource

        self.source = source or AlpacaSource(**({"base_url": base_url} if base_url else {}))

    def fetch(self, symbols, start, end):
        return {symbol: self.source.fetch(symbol, "1d", start, end).set_index("timestamp")
                .loc[:, ["open", "high", "low", "close", "volume"]]
                for symbol in symbols}


def update_market_bars(session, symbols, client, start=None, dry_run=False, snapshot=None):
    """Refresh selected symbols and return their latest-session status and row counts."""
    if snapshot:
        before = {symbol: present_sessions(session, symbol) for symbol in symbols}
        missing = backfill_snapshot(session, snapshot)
        results = {symbol: {"rows": len(present_sessions(session, symbol) - before[symbol]),
                            "missing_snapshot": symbol in missing,
                            "status": market_bar_status(present_sessions(session, symbol), datetime.now(timezone.utc))}
                   for symbol in symbols}
        if dry_run:
            session.rollback()
        else:
            session.commit()
        return results
    now = datetime.now(timezone.utc)
    ranges = {}
    for symbol in symbols:
        latest, _, missing = market_bar_status(present_sessions(session, symbol), now)
        ranges[symbol] = (start or (min(missing) if missing else latest), latest)
    frames = client.fetch(symbols, min(a for a, _ in ranges.values()), max(b for _, b in ranges.values()))
    results = {}
    for symbol in symbols:
        frame = frames.get(symbol)
        count = 0 if frame is None else len(frame)
        if frame is not None and not dry_run:
            upsert_bars(session, symbol, frame, "alpaca")
        results[symbol] = {"rows": count, "status": market_bar_status(
            present_sessions(session, symbol) if not dry_run else present_sessions(session, symbol) | {
                idx.date() if hasattr(idx, "date") else idx for idx in (frame.index if frame is not None else [])
            }, now)}
    if not dry_run:
        session.commit()
    return results
