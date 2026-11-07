from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from apps.api.app import app, get_db
from quantagent.database import Base
from quantagent.models import MarketBar


def test_market_series_and_candles_contract(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'market.db'}")
    Base.metadata.create_all(engine, tables=[MarketBar.__table__])
    with Session(engine) as db:
        for day, close in ((date(2024, 1, 1), 10), (date(2024, 2, 1), 11),
                           (date(2024, 3, 1), 12), (date(2024, 4, 1), 13)):
            db.add(MarketBar(symbol="SPY", session_date=day, open=close - 1, high=close + 1,
                             low=close - 2, close=close, volume=100, source="test"))
        db.commit()
        app.dependency_overrides[get_db] = lambda: db
        client = TestClient(app)
        try:
            response = client.get("/api/market/series?symbols=SPY,QQQ&range=1m")
            assert response.status_code == 200
            assert response.json() == [
                {"symbol": "SPY", "points": [{"date": "2024-03-01", "value": 12.0},
                                                {"date": "2024-04-01", "value": 13.0}]},
                {"symbol": "QQQ", "points": []},
            ]
            candles = client.get("/api/market/etf/SPY/candles?range=3m")
            assert candles.status_code == 200
            assert set(candles.json()[0]) == {"date", "open", "high", "low", "close"}
            assert candles.json() == [
                {"date": "2024-01-01", "open": 9.0, "high": 11.0, "low": 8.0, "close": 10.0},
                {"date": "2024-02-01", "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0},
                {"date": "2024-03-01", "open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0},
                {"date": "2024-04-01", "open": 12.0, "high": 14.0, "low": 11.0, "close": 13.0},
            ]
            assert set(response.json()[0]) == {"symbol", "points"}
            assert set(response.json()[0]["points"][0]) == {"date", "value"}
            assert client.get("/api/market/series?symbols=SPY,QQQ,IWM,XLB,XLE,XLF&range=1m").status_code == 422
            assert client.get("/api/market/series?symbols=NOPE&range=1m").status_code == 422
            assert client.get("/api/market/series?symbols=SPY&range=6m").status_code == 422
            assert client.get("/api/market/etf/NOPE/candles?range=1y").status_code == 404
            assert client.get("/api/market/etf/SPY/candles?range=1m").status_code == 422
            assert client.get("/api/market/etf/QQQ/candles?range=3m").json() == []
        finally:
            app.dependency_overrides.clear()
