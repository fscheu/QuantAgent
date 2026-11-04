from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.market_etfs import ETF_SYMBOLS, filas_etf, held_in_por_simbolo
from quantagent.database import Base
from quantagent.models import MarketBar


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    with Session(engine) as db:
        yield db


def put_bar(session, symbol, day, close):
    session.add(MarketBar(symbol=symbol, session_date=day, open=close, high=close, low=close, close=close, volume=1, source="test"))


def test_returns_spark_and_held_in(session):
    start = date(2024, 10, 1)
    for i in range(64):
        put_bar(session, "SPY", start + timedelta(days=i), 100 + i)
    session.commit()
    row = next(row for row in filas_etf(session, {"SPY": ["paper-1"]}) if row["symbol"] == "SPY")
    # Ventanas: 1, 5, 20 y 62 sesiones atrás (marketConfig.minSessions = ventana + 1).
    assert row["returns"] == pytest.approx({"d1": (163 / 162 - 1) * 100, "w1": (163 / 158 - 1) * 100, "m1": (163 / 143 - 1) * 100, "m3": (163 / 101 - 1) * 100, "ytd": None})
    assert len(row["spark_30d"]) == 30
    assert row["held_in"] == ["paper-1"]
    assert held_in_por_simbolo([("paper-1", ["SPY", "QQQ"]), ("paper-2", ["SPY"])]) == {"SPY": ["paper-1", "paper-2"], "QQQ": ["paper-1"]}


def test_insufficient_history_and_missing_symbol(session):
    put_bar(session, "SPY", date(2025, 1, 2), 10)
    session.commit()
    rows = {row["symbol"]: row for row in filas_etf(session)}
    assert len(rows) == 29 and set(rows) == set(ETF_SYMBOLS)
    assert rows["SPY"]["returns"] == {"d1": None, "w1": None, "m1": None, "m3": None, "ytd": None}
    assert rows["SPY"]["history_sessions"] == 1
    assert rows["QQQ"]["close"] is None and rows["QQQ"]["returns"] == {key: None for key in ("d1", "w1", "m1", "m3", "ytd")}
