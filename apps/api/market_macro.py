"""Macro series exposed by the market endpoint."""

from datetime import date, timedelta
from math import ceil

from sqlalchemy import select

from quantagent.models import MacroObservation

# Mirror marketConfig.macroStale in docs/03_design/web-design-system/config.ts.
MACRO_STALE_GRACE_BUSINESS_DAYS = 2
MACRO_FALLBACK_FREQUENCY_MULTIPLIER = 1.5

_SERIES = (
    ("T10Y2Y", "Curva del Tesoro", "daily", "pp"),
    ("CPIAUCSL", "Inflación", "monthly", "index"),
    ("UNRATE", "Desempleo", "monthly", "pct"),
    ("PAYEMS", "Empleo no agrícola", "monthly", "index"),
    ("INDPRO", "Producción industrial", "monthly", "index"),
    ("BAMLH0A0HYM2", "High yield OAS", "daily", "pct"),
    ("VIXCLS", "VIX", "daily", "index"),
    ("T5YIE", "Inflación esperada a 5 años", "daily", "pct"),
    ("DFII10", "Tasa real a 10 años", "daily", "pct"),
    ("DTWEXBGS", "Dólar amplio", "daily", "index"),
)


def series_macro(db, today=None, expected_release=None):
    today = today or date.today()
    data = db.scalars(
        select(MacroObservation)
        .where(MacroObservation.series_id == "T10Y2Y")
        .order_by(MacroObservation.observation_date)
    ).all() if db is not None else []
    active = {
        "T10Y2Y": {
            "last": None,
            "change_1m": None,
            "stale": False,
            "expected_release": expected_release.isoformat() if expected_release else None,
            "points": [],
        }
    }
    if data:
        last = data[-1]
        month_before = last.observation_date - timedelta(days=30)
        prior = [row for row in data if row.observation_date <= month_before]
        stale_anchor = expected_release or last.observation_date
        business_days = sum(
            (stale_anchor + timedelta(days=offset)).weekday() < 5
            for offset in range(1, (today - stale_anchor).days + 1)
        )
        stale_limit = MACRO_STALE_GRACE_BUSINESS_DAYS
        if expected_release is None:
            stale_limit += ceil(MACRO_FALLBACK_FREQUENCY_MULTIPLIER)
        active["T10Y2Y"] = {
            "last": {"value": float(last.value), "observation_date": last.observation_date.isoformat(),
                     "published_at": last.published_at.isoformat()},
            "change_1m": float(last.value - prior[-1].value) if prior else None,
            "stale": business_days > stale_limit,
            "expected_release": expected_release.isoformat() if expected_release else None,
            "points": [{"date": row.observation_date.isoformat(), "value": float(row.value)}
                       for row in data if row.observation_date >= today - timedelta(days=365)],
        }
    return [
        {
            "code": code,
            "title": title,
            "frequency": frequency,
            "status": "active" if code == "T10Y2Y" else "coming_soon",
            "unit": unit,
            **active.get(code, {"last": None, "change_1m": None, "stale": False,
                                "expected_release": None, "points": []}),
        }
        for code, title, frequency, unit in _SERIES
    ]


def _fin_de_mes(day):
    """USREC trae un dato por mes fechado el día 1: el período cerrado llega hasta el último día de ese mes."""
    return (day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


def recession_periods(db):
    rows = db.scalars(
        select(MacroObservation)
        .where(MacroObservation.series_id == "USREC")
        .order_by(MacroObservation.observation_date)
    ).all()
    periods = []
    start = None
    last = None
    for row in rows:
        if row.value == 1:
            if start is None:
                start = row.observation_date
            last = row.observation_date
        elif start is not None:
            periods.append({"start": start.isoformat(), "end": _fin_de_mes(last).isoformat()})
            start = None
    if start is not None:
        periods.append({"start": start.isoformat(), "end": last.isoformat()})
    return periods
