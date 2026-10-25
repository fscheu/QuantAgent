"""Sesiones cerradas de mercado para ejecución paper de carteras."""

from datetime import datetime

import pandas as pd
import pandas_market_calendars as mcal

from quantagent.backtesting.signals import bar_signals
from quantagent.fabrica.transversal import METRICAS, es_transversal, señales_transversales
from quantagent.strategy.registry import STRATEGY_REGISTRY, build_strategy, cli_aliases

_NYSE = mcal.get_calendar("NYSE")


def _a_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def sesiones_cerradas_nyse(
    as_of: datetime | pd.Timestamp,
    n_sesiones: int = 253,
) -> pd.DatetimeIndex:
    """Índice de sesiones NYSE cerradas al momento `as_of` (UTC inyectado)."""
    ts_utc = _a_utc(as_of)
    dias = max(int(n_sesiones * 2) + 30, 10)
    sched = _NYSE.schedule(start_date=(ts_utc - pd.Timedelta(days=dias)).date(), end_date=ts_utc.date())
    cerradas = sched[sched["market_close"] <= ts_utc]
    return cerradas.index[-n_sesiones:]


def ultima_sesion_cerrada_nyse(as_of: datetime | pd.Timestamp) -> pd.Timestamp:
    """Fecha normalizada de la última rueda NYSE cerrada al momento `as_of`."""
    return sesiones_cerradas_nyse(as_of, n_sesiones=1)[-1]


def entradas_cartera(h, proveedor, reloj, posiciones=None):
    """Devuelve (fecha_cerrada, panel_universo, mapa_precios) o None si faltan datos."""
    as_of = reloj()
    sesiones = sesiones_cerradas_nyse(as_of, n_sesiones=504)
    ultima = sesiones[-1]
    universo = sorted({a for p in h.piernas for a in p.activos})
    if not universo:
        return None
    held_extra = set(posiciones or ()) - set(universo)
    inicio, fin = sesiones[0].to_pydatetime(), (ultima + pd.Timedelta(days=1)).to_pydatetime()
    series = {}
    for s in universo:
        df = proveedor.get_ohlc(s, "1d", inicio, fin)
        if df.empty:
            return None
        fechas = pd.to_datetime(df["timestamp"]).dt.normalize()
        c = df.loc[fechas.isin(sesiones), "close"]
        c.index = fechas[fechas.isin(sesiones)]
        series[s] = c[~c.index.duplicated(keep="last")]
    panel = pd.DataFrame(series).reindex(sesiones)
    n_validos = int((~panel.isna().any(axis=1))[::-1].cumprod().sum())
    if n_validos < 253:
        return None
    panel = panel.iloc[-n_validos:]
    precios = {s: float(panel.loc[ultima, s]) for s in universo}
    for s in held_extra:
        df = proveedor.get_ohlc(s, "1d", ultima.to_pydatetime(), fin)
        if df.empty:
            return None
        c = df.loc[pd.to_datetime(df["timestamp"]).dt.normalize() == ultima, "close"]
        if c.empty or pd.isna(c.iloc[-1]):
            return None
        precios[s] = float(c.iloc[-1])
    return ultima, panel, precios


def exposicion_piernas(h, panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula exposición binaria (1/0) por (pierna_idx, activo) para panel de cierres."""
    aliases, tf = cli_aliases(), getattr(h, "timeframe", "1d")
    cols = [(i, a) for i, p in enumerate(h.piernas) for a in (getattr(p, "activos", None) or ()) if a in panel.columns]
    expos = pd.DataFrame(0, index=panel.index, columns=pd.MultiIndex.from_tuples(cols), dtype=int)
    for i, p in enumerate(h.piernas):
        clase = getattr(p, "clase", None) or (p.get("clase") if isinstance(p, dict) else None)
        activos = [a for a in (getattr(p, "activos", None) or ()) if a in panel.columns]
        if not activos:
            continue
        if es_transversal(clase):
            sel = señales_transversales(panel[activos], METRICAS[clase])
            for a in activos:
                expos[(i, a)] = sel[a].values
            continue
        reg = aliases.get(clase, clase)
        if reg not in STRATEGY_REGISTRY:
            continue
        raw = getattr(p, "parametros", None) or {}
        params = p.combinaciones()[0] if hasattr(p, "combinaciones") and p.combinaciones() else {k: v[0] if isinstance(v, list) else v for k, v in raw.items()}
        strat = build_strategy(reg, **params)
        for a in activos:
            c = panel[a]
            candles = pd.DataFrame({"timestamp": panel.index, "open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=panel.index)
            sig = bar_signals(strat, candles, a, tf)
            pos, estados = 0, []
            for e, x in zip(sig["entry"], sig["exit"]):
                pos = 1 if e == 1 else (0 if x else pos)
                estados.append(pos)
            expos[(i, a)] = estados
    return expos
