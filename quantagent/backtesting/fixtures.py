"""Loader for versioned OHLCV fixtures used by the backtest CLI (D10 of PLAN-30-DIAS)."""

import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from quantagent.models import MarketData

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


@dataclass
class FixtureMetadata:
    """Symbol/timeframe/date-range summary of a fixture, used to size a Backtest run."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    row_count: int


def fixture_metadata(name: str) -> FixtureMetadata:
    """Read `tests/fixtures/<name>.csv` and summarize it without seeding the DB."""
    path = FIXTURES_DIR / f"{name}.csv"
    symbol = None
    timeframe = None
    start_date = None
    end_date = None
    row_count = 0

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            symbol = symbol or row["symbol"]
            timeframe = timeframe or row["timeframe"]
            timestamp = datetime.fromisoformat(row["timestamp"])
            start_date = timestamp if start_date is None else min(start_date, timestamp)
            end_date = timestamp if end_date is None else max(end_date, timestamp)
            row_count += 1

    if row_count == 0:
        raise ValueError(f"Fixture '{name}' has no rows: {path}")

    return FixtureMetadata(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        row_count=row_count,
    )


def load_fixture(session: Session, name: str) -> int:
    """Seed `market_data` from `tests/fixtures/<name>.csv`. Returns the row count."""
    path = FIXTURES_DIR / f"{name}.csv"
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                MarketData(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=Decimal(row["volume"]),
                )
            )

    session.bulk_save_objects(rows)
    session.commit()
    return len(rows)
