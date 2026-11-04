"""Lectura del estado semanal M5 y retornos de los ETFs de mercado."""

import os
from pathlib import Path

import pandas as pd

from quantagent.fabrica.screener import ETF_SECTOR
from quantagent.models import MarketBar
from apps.api.market_etfs import retornos_simbolo

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


def filas_sector(session, carpeta=None, held_in=None):
    row = _regimen(carpeta)
    enabled = set(str(row.habilitados).split(",")) if row is not None else set()
    blocked = set(str(row.bloqueados).split(",")) if row is not None else set()
    scores = {s: float(row[f"score_{s}"]) for s in SECTORS} if row is not None else {}
    ranked = {s: i + 1 for i, s in enumerate(sorted(
        (s for s in SECTORS if s in enabled), key=lambda s: scores[s], reverse=True))}
    held_in = held_in or {}
    result = []
    for symbol in UNIVERSE:
        sector = symbol in SECTORS
        score = scores.get(symbol) if row is not None else None
        is_blocked = (symbol in blocked or (sector and symbol not in enabled)) if row is not None else False
        result.append({"symbol": symbol, "name": SECTORS.get(symbol, REFERENCES.get(symbol)),
            "is_reference": not sector, "held_in": held_in.get(symbol, []),
            "returns": retornos_simbolo(session, symbol)[0] if session is not None else
                       {k: None for k in ("d1", "w1", "m1", "m3", "ytd")},
            "momentum_score": score, "rank": ranked.get(symbol), "enabled": symbol in enabled,
            "blocked": is_blocked, "blocked_reason": ("filtro de caída" if symbol in blocked else
                        "score no positivo" if row is not None and sector and score <= 0 else
                        "fuera del top 3" if is_blocked and sector else None),
            "score_as_of": row.fecha.date().isoformat() if row is not None else None,
            "state_description": None})
    return result
