"""Loader for versioned OHLCV fixtures used by the backtest CLI (D10 of PLAN-30-DIAS)."""

import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy.orm import Session

from quantagent.models import MarketData

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


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
