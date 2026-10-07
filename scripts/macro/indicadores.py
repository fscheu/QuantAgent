#!/usr/bin/env python
"""Fetch the weekly macro dashboard indicators from direct sources (QuantAgent-577).

Usage: python scripts/macro/indicadores.py [--as-of YYYY-MM-DD]
Prints one JSON document: for every indicator its value at the week close (last observation up to
the last Friday), the same one week earlier, the change, the latest observation and the source.
The weekly macro note takes its numbers from here; the model that writes the note never computes
them. Imports nothing from `quantagent`. An indicator that fails keeps its error and the run goes on.
"""

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from datetime import date, timedelta

TIMEOUT = 20
RETRIES = 3
# FRED stalls requests that announce themselves as a bare "Mozilla/5.0"; an honest client name gets through.
USER_AGENT = "QuantAgent-macro/1.0 (+https://github.com/fscheu/QuantAgent)"


def week_close(as_of):
    """Last Friday on or before `as_of`: the close the weekly note reports on."""
    return as_of - timedelta(days=(as_of.weekday() - 4) % 7)


def last_on_or_before(series, day):
    """Latest (date, value) of a date-sorted series that is not after `day`, or None."""
    found = None
    for obs in series:
        if obs[0] > day:
            break
        found = obs
    return found


def change(kind, value, prev):
    """`pct` for prices (percent), `abs` for rates and spreads (same unit as the value)."""
    if kind == "pct":
        return (value / prev - 1) * 100
    return value - prev


def weekly(series, as_of, kind):
    """Week close, the close one week before and the change between them."""
    close_day = week_close(as_of)
    close = last_on_or_before(series, close_day)
    prev = last_on_or_before(series, close_day - timedelta(days=7))
    latest = last_on_or_before(series, as_of)
    if close is None or latest is None:
        raise ValueError("no observations up to the week close")
    out = {
        "value": round(close[1], 4),
        "date": close[0].isoformat(),
        "latest_value": round(latest[1], 4),
        "latest_date": latest[0].isoformat(),
        "prev_value": None,
        "prev_date": None,
        "change": None,
    }
    # Same observation on both sides means the source has no new data for the week: no change to report.
    if prev is not None and prev[0] != close[0]:
        out.update(prev_value=round(prev[1], 4), prev_date=prev[0].isoformat())
        out["change"] = round(change(kind, close[1], prev[1]), 4)
    return out


def yoy(series):
    """Year-over-year percent change of a monthly index: [(month, yoy_pct), ...] for months with a year behind."""
    by_month = {(d.year, d.month): v for d, v in series}
    return [
        (d, (v / by_month[(d.year - 1, d.month)] - 1) * 100) for d, v in series if (d.year - 1, d.month) in by_month
    ]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8")
        except OSError:  # timeouts and resets come and go on these public endpoints
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))


def fred(series_id, start):
    """Daily or monthly series from FRED's public CSV export (no API key)."""
    text = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start.isoformat()}")
    rows = list(csv.reader(io.StringIO(text)))[1:]
    return [(date.fromisoformat(d), float(v)) for d, v in rows if v not in ("", ".")]  # "." marks a holiday


def yahoo(ticker, start):
    import yfinance as yf  # imported here so the pure helpers above can be tested without it

    closes = yf.Ticker(ticker).history(start=start.isoformat(), interval="1d")["Close"].dropna()
    return [(ts.date(), float(v)) for ts, v in closes.items()]


def bcra_usd(start, end):
    """BCRA reference closing rate for the US dollar."""
    url = (
        "https://api.bcra.gob.ar/estadisticascambiarias/v1.0/Cotizaciones/USD"
        f"?fechadesde={start.isoformat()}&fechahasta={end.isoformat()}&limit=100"
    )
    rows = json.loads(_get(url))["results"]
    return sorted((date.fromisoformat(r["fecha"]), float(r["detalle"][0]["tipoCotizacion"])) for r in rows)


def argentinadatos(path, key, start):
    rows = json.loads(_get(f"https://api.argentinadatos.com/v1/{path}"))
    series = sorted((date.fromisoformat(r["fecha"]), float(r[key])) for r in rows)
    return [obs for obs in series if obs[0] >= start]


FRED_URL = "https://fred.stlouisfed.org/series/"
YAHOO_URL = "https://finance.yahoo.com/quote/"

# id, name, unit, change kind, source name, source url, official source?, fetcher(start, end)
INDICATORS = [
    ("fed_funds", "Tasa de la Fed (techo del rango)", "%", "abs", "Reserva Federal vía FRED", FRED_URL + "DFEDTARU",
     True, lambda s, e: fred("DFEDTARU", s)),
    ("us10y", "Bono de EE.UU. a 10 años", "%", "abs", "Tesoro de EE.UU. vía FRED", FRED_URL + "DGS10",
     True, lambda s, e: fred("DGS10", s)),
    ("dxy", "Dólar global (DXY)", "puntos", "pct", "Yahoo Finance", YAHOO_URL + "DX-Y.NYB",
     False, lambda s, e: yahoo("DX-Y.NYB", s)),
    ("sp500", "S&P 500", "puntos", "pct", "S&P Dow Jones vía FRED", FRED_URL + "SP500",
     True, lambda s, e: fred("SP500", s)),
    ("nasdaq", "Nasdaq Composite", "puntos", "pct", "Nasdaq vía FRED", FRED_URL + "NASDAQCOM",
     True, lambda s, e: fred("NASDAQCOM", s)),
    ("stoxx600", "Europa (Stoxx 600)", "puntos", "pct", "Yahoo Finance", YAHOO_URL + "%5ESTOXX",
     False, lambda s, e: yahoo("^STOXX", s)),
    ("nikkei", "Japón (Nikkei 225)", "puntos", "pct", "Yahoo Finance", YAHOO_URL + "%5EN225",
     False, lambda s, e: yahoo("^N225", s)),
    ("emergentes", "Emergentes (ETF EEM, sigue al MSCI EM)", "USD", "pct", "Yahoo Finance", YAHOO_URL + "EEM",
     False, lambda s, e: yahoo("EEM", s)),
    ("brent", "Petróleo Brent (futuro más cercano)", "USD", "pct", "Yahoo Finance", YAHOO_URL + "BZ%3DF",
     False, lambda s, e: yahoo("BZ=F", s)),
    ("oro", "Oro (futuro más cercano)", "USD", "pct", "Yahoo Finance", YAHOO_URL + "GC%3DF",
     False, lambda s, e: yahoo("GC=F", s)),
    ("vix", "VIX", "puntos", "abs", "CBOE vía FRED", FRED_URL + "VIXCLS",
     True, lambda s, e: fred("VIXCLS", s)),
    ("hy_spread", "Spread de bonos de alto rendimiento", "pp", "abs", "ICE BofA vía FRED", FRED_URL + "BAMLH0A0HYM2",
     True, lambda s, e: fred("BAMLH0A0HYM2", s)),
    ("ar_usd_oficial", "Argentina: dólar oficial (referencia BCRA)", "ARS", "pct", "BCRA",
     "https://www.bcra.gob.ar/PublicacionesEstadisticas/Tipos_de_cambios.asp", True, bcra_usd),
    ("ar_ccl", "Argentina: dólar CCL", "ARS", "pct", "argentinadatos.com (no oficial)",
     "https://argentinadatos.com/", False,
     lambda s, e: argentinadatos("cotizaciones/dolares/contadoconliqui", "venta", s)),
    ("ar_riesgo_pais", "Argentina: riesgo país", "pb", "abs", "argentinadatos.com (no oficial)",
     "https://argentinadatos.com/", False, lambda s, e: argentinadatos("finanzas/indices/riesgo-pais", "valor", s)),
    ("ar_merval", "Argentina: Merval (en pesos)", "puntos", "pct", "Yahoo Finance", YAHOO_URL + "%5EMERV",
     False, lambda s, e: yahoo("^MERV", s)),
]


def _entry(ind_id, name, unit, kind, source, url, official):
    return {"id": ind_id, "name": name, "unit": unit, "change_kind": kind, "source": source, "source_url": url,
            "official": official}


def collect(as_of):
    start = as_of - timedelta(days=30)
    out = []
    for ind_id, name, unit, kind, source, url, official, fetch in INDICATORS:
        entry = _entry(ind_id, name, unit, kind, source, url, official)
        try:
            entry.update(weekly(fetch(start, as_of), as_of, kind))
        except Exception as exc:  # one dead source must not take the dashboard down
            entry["error"] = f"{type(exc).__name__}: {exc}"
        out.append(entry)

    # US inflation is monthly: latest year-over-year reading and the one before, no weekly change.
    entry = _entry("us_cpi_yoy", "Inflación de EE.UU. (IPC, anual)", "%", "abs", "BLS vía FRED",
                   FRED_URL + "CPIAUCSL", True)
    try:
        readings = yoy(fred("CPIAUCSL", as_of - timedelta(days=460)))
        (prev_d, prev_v), (d, v) = readings[-2], readings[-1]
        entry.update(value=round(v, 2), date=d.isoformat(), prev_value=round(prev_v, 2), prev_date=prev_d.isoformat(),
                     change=round(v - prev_v, 2), frequency="monthly")
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    out.append(entry)

    # The gap is derived: how far the CCL sits above the official rate, in percent.
    by_id = {e["id"]: e for e in out}
    ccl, oficial = by_id["ar_ccl"], by_id["ar_usd_oficial"]
    entry = _entry("ar_brecha", "Argentina: brecha CCL / oficial", "%", "abs", "calculado (CCL / oficial - 1)", "",
                   False)
    if "error" in ccl or "error" in oficial:
        entry["error"] = "needs ar_ccl and ar_usd_oficial"
    else:
        entry.update(value=round((ccl["value"] / oficial["value"] - 1) * 100, 2), date=ccl["date"])
        if ccl["prev_value"] and oficial["prev_value"]:
            prev = (ccl["prev_value"] / oficial["prev_value"] - 1) * 100
            entry.update(prev_value=round(prev, 2), prev_date=ccl["prev_date"],
                         change=round(entry["value"] - prev, 2))
    out.append(entry)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)
    indicators = collect(args.as_of)
    doc = {
        "as_of": args.as_of.isoformat(),
        "week_close": week_close(args.as_of).isoformat(),
        "failed": [e["id"] for e in indicators if "error" in e],
        "indicators": indicators,
    }
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 1 if len(doc["failed"]) == len(indicators) else 0


if __name__ == "__main__":
    sys.exit(main())
