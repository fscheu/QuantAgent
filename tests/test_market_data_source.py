"""Source uniqueness and the real Alembic round trip on PostgreSQL."""

import os
import subprocess
import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from quantagent.models import MarketData


def test_source_uniqueness(db_session):
    candle = dict(symbol="SPY", timeframe="1d", timestamp=datetime(2020, 1, 1),
                  open=1, high=1, low=1, close=1, volume=1)
    db_session.add_all([MarketData(**candle), MarketData(**candle, source="alpaca")])
    db_session.commit()
    assert {row.source for row in db_session.query(MarketData)} == {"legacy", "alpaca"}
    db_session.add(MarketData(**candle, source="legacy"))
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_postgres_migration_round_trip():
    url = os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("PostgreSQL required for migration acceptance")
    engine = create_engine(url)
    schema = "source_test_" + uuid.uuid4().hex
    env = dict(os.environ, PGOPTIONS=f"-c search_path={schema}")
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            conn.execute(text("CREATE TABLE market_data (id INTEGER PRIMARY KEY, "
                              "symbol TEXT, timeframe TEXT, timestamp TIMESTAMP, close NUMERIC)"))
            conn.execute(text("INSERT INTO market_data VALUES "
                              "(1, 'SPY', '1d', '2020-01-01', 10), "
                              "(2, 'SPY', '1d', '2020-01-01', 20)"))
        for args in [("stamp", "25efa4fd0a2b"), ("upgrade", "head")]:
            subprocess.run(["python", "-m", "alembic", *args], env=env, check=True)
        with engine.begin() as conn:
            conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            assert conn.execute(text("SELECT id, source, close FROM market_data")).one() == (1, "legacy", 10)
            conn.execute(text("INSERT INTO market_data VALUES (3, 'SPY', '1d', '2020-01-01', 5, 'alpaca')"))
            with pytest.raises(IntegrityError), conn.begin_nested():
                conn.execute(text("INSERT INTO market_data VALUES (4, 'SPY', '1d', '2020-01-01', 5, 'alpaca')"))
        subprocess.run(["python", "-m", "alembic", "downgrade", "-1"], env=env, check=True)
        with engine.connect() as conn:
            assert conn.execute(text(f'SELECT count(*) FROM "{schema}".market_data')).scalar() == 2
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        engine.dispose()
