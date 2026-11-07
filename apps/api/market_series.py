"""On-demand ETF comparison series and daily candles."""

from calendar import monthrange

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantagent.data.market_bars import MARKET_BAR_SYMBOLS
from quantagent.models import MarketBar

from .app import get_db
from .schemas import DailyCandle, MarketSeries

router = APIRouter()
SERIES_RANGES = {"1m": 1, "3m": 3, "1y": 12}
CANDLE_RANGES = {"3m": 3, "6m": 6, "1y": 12}


def _cutoff(latest, months):
    month = latest.month - months
    year = latest.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(latest.day, monthrange(year, month)[1])
    return latest.replace(year=year, month=month, day=day)


def _bars(db, symbol, range_name, ranges):
    if symbol not in MARKET_BAR_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Símbolo desconocido: {symbol}")
    if range_name not in ranges:
        raise HTTPException(status_code=422, detail=f"Rango inválido: {range_name}")
    rows = db.scalars(select(MarketBar).where(MarketBar.symbol == symbol)
                      .order_by(MarketBar.session_date)).all()
    if not rows:
        return []
    cutoff = _cutoff(rows[-1].session_date, ranges[range_name])
    return [row for row in rows if row.session_date >= cutoff]


@router.get("/api/market/series", response_model=list[MarketSeries])
def market_series(symbols: str = Query(...), range: str = Query(...), db: Session = Depends(get_db)):
    requested = symbols.split(",")
    if len(requested) > 5:
        raise HTTPException(status_code=422, detail="Se permiten como máximo 5 símbolos")
    if range not in SERIES_RANGES:
        raise HTTPException(status_code=422, detail=f"Rango inválido: {range}")
    if any(symbol not in MARKET_BAR_SYMBOLS for symbol in requested):
        invalid = next(symbol for symbol in requested if symbol not in MARKET_BAR_SYMBOLS)
        raise HTTPException(status_code=422, detail=f"Símbolo desconocido: {invalid}")
    result = []
    for symbol in requested:
        rows = _bars(db, symbol, range, SERIES_RANGES)
        result.append({"symbol": symbol, "points": [
            {"date": row.session_date.isoformat(), "value": float(row.close)} for row in rows
        ]})
    return result


@router.get("/api/market/etf/{symbol}/candles", response_model=list[DailyCandle])
def market_candles(symbol: str, range: str = Query(...), db: Session = Depends(get_db)):
    if symbol not in MARKET_BAR_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Símbolo desconocido: {symbol}")
    if range not in CANDLE_RANGES:
        raise HTTPException(status_code=422, detail=f"Rango inválido: {range}")
    return [{"date": row.session_date.isoformat(), "open": float(row.open), "high": float(row.high),
             "low": float(row.low), "close": float(row.close)}
            for row in _bars(db, symbol, range, CANDLE_RANGES)]
