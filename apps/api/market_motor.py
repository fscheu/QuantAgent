"""Lectura del estado semanal M5 y retornos de los ETFs de mercado."""

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quantagent.fabrica.screener import ETF_SECTOR
from quantagent.models import MarketBar

SECTORS = ETF_SECTOR
REFERENCES = {"SPY": "S&P 500", "EEM": "Emerging Markets", "EWZ": "Brazil"}
UNIVERSE = (*SECTORS, *REFERENCES)


def _path(carpeta):
    return Path(carpeta or os.getenv("QUANTAGENT_M5_DIR", "local-data")) / "m5-regimen-semanal.csv"


def _regimen(carpeta):
    path = _path(carpeta)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"fecha", "estado", "habilitados", "bloqueados", *(f"score_{s}" for s in UNIVERSE)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Formato inválido en {path}: faltan columnas {', '.join(sorted(missing))}")
    if frame.empty:
        return None
    frame["fecha"] = pd.to_datetime(frame.fecha, errors="raise")
    return frame.sort_values("fecha").iloc[-1]


def estado_motor(carpeta=None):
    row = _regimen(carpeta)
    if row is None:
        return None
    state = str(row.estado).replace("-", "_")
    return {"state": state, "computed_at": row.fecha.isoformat(),
            "next_at": (row.fecha + pd.Timedelta(days=7)).isoformat(),
            "state_description": None}


def _returns(session, symbol, as_of):
    start = as_of - timedelta(days=400)
    bars = session.query(MarketBar.session_date, MarketBar.close).filter(
        MarketBar.symbol == symbol, MarketBar.session_date >= start,
        MarketBar.session_date <= as_of).order_by(MarketBar.session_date).all()
    closes = pd.Series([float(b.close) for b in bars], index=[b.session_date for b in bars])
    result = {}
    for key, sessions in (("d1", 1), ("w1", 5), ("m1", 21), ("m3", 63), ("ytd", None)):
        subset = closes[closes.index >= date(as_of.year, 1, 1)] if key == "ytd" else closes
        n = len(subset) - 1 if sessions is None else sessions
        result[key] = (float(subset.iloc[-1] / subset.iloc[-n - 1] - 1)
                       if n > 0 and len(subset) > n and subset.iloc[-n - 1] else None)
    return result


def filas_sector(session, carpeta=None, held_in=None):
    row = _regimen(carpeta)
    enabled = set(str(row.habilitados).split(",")) if row is not None else set()
    blocked = set(str(row.bloqueados).split(",")) if row is not None else set()
    scores = {s: float(row[f"score_{s}"]) for s in SECTORS} if row is not None else {}
    ranked = {s: i + 1 for i, s in enumerate(sorted(
        (s for s in SECTORS if s in enabled), key=lambda s: scores[s], reverse=True))}
    held_in = held_in or {}
    result = []
    latest = (session.query(MarketBar.session_date).filter(
        MarketBar.symbol.in_(UNIVERSE)).order_by(MarketBar.session_date.desc()).first()
        if session is not None else None)
    as_of = row.fecha.date() if row is not None else (latest[0] if latest else None)
    for symbol in UNIVERSE:
        sector = symbol in SECTORS
        score = scores.get(symbol) if row is not None else None
        is_blocked = (symbol in blocked or (sector and symbol not in enabled)) if row is not None else False
        result.append({"symbol": symbol, "name": SECTORS.get(symbol, REFERENCES.get(symbol)),
            "is_reference": not sector, "held_in": held_in.get(symbol, []),
            "returns": _returns(session, symbol, as_of) if as_of is not None else
                       {k: None for k in ("d1", "w1", "m1", "m3", "ytd")},
            "momentum_score": score, "rank": ranked.get(symbol), "enabled": symbol in enabled,
            "blocked": is_blocked, "blocked_reason": ("filtro de caída" if symbol in blocked else
                        "score no positivo" if row is not None and sector and score <= 0 else
                        "fuera del top 3" if is_blocked and sector else None),
            "score_as_of": row.fecha.date().isoformat() if row is not None else None,
            "state_description": None})
    return result
