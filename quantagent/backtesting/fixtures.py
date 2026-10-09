"""Loader for versioned OHLCV fixtures used by the backtest CLI (D10 of PLAN-30-DIAS)."""

import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from quantagent.data.snapshot import load_manifest, read_snapshot
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


def _snapshot_range(name: str, symbol: str, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """Velas ajustadas de `symbol` entre `start` y `end` inclusive (fechas); `None` deja abierto ese borde."""
    df = read_snapshot(name, symbol)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if df.empty:
        raise ValueError(f"El snapshot '{name}' no tiene velas de {symbol} en el rango pedido")
    return df


def snapshot_metadata(name: str, symbol: str, start: Optional[datetime], end: Optional[datetime]) -> FixtureMetadata:
    """Resume el rango de un snapshot como `fixture_metadata`, sin tocar la base."""
    df = _snapshot_range(name, symbol, start, end)
    return FixtureMetadata(
        symbol=symbol,
        timeframe=load_manifest(name)["timeframe"],
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        row_count=len(df),
    )


def load_snapshot(session: Session, name: str, symbol: str, start: Optional[datetime], end: Optional[datetime]) -> int:
    """Carga en `market_data` las velas ajustadas del rango, como `load_fixture`. Devuelve la cantidad de filas."""
    df = _snapshot_range(name, symbol, start, end)
    timeframe = load_manifest(name)["timeframe"]
    rows = [
        MarketData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=ts.to_pydatetime(),
            open=Decimal(f"{r.open:.8f}"),
            high=Decimal(f"{r.high:.8f}"),
            low=Decimal(f"{r.low:.8f}"),
            close=Decimal(f"{r.close:.8f}"),
            volume=Decimal(int(r.volume)),
        )
        for ts, r in df.iterrows()
    ]
    session.bulk_save_objects(rows)
    session.commit()
    return len(rows)
