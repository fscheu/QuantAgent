"""Medición de Alpaca con claves reales (QuantAgent-v38v). Solo GET de datos; no guarda velas."""

import sys
from time import perf_counter

import pandas as pd
import requests

from quantagent.backtesting.signals import bar_signals
from quantagent.data.sources.alpaca import BARS_PATH, DATA_HOST, TIMEFRAMES, AlpacaSource
from quantagent.strategy.registry import build_strategy, cli_aliases

SYMBOL = "SPY"
START = pd.Timestamp("2016-01-04", tz="UTC")
LIMIT_HEADERS = ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset")


def pedir(source, headers, timeframe, feed="sip", adjustment="split", start=START, limit=10000):
    """Un pedido paginado; devuelve (velas, segundos, pedidos, headers de límite del primero)."""
    end = pd.Timestamp(source.latest())
    params = {"symbols": SYMBOL, "timeframe": TIMEFRAMES[timeframe], "start": start.isoformat(),
              "end": end.isoformat(), "feed": feed, "adjustment": adjustment, "limit": limit}
    bars, calls, first, t0 = [], 0, {}, perf_counter()
    while True:
        resp = requests.get(DATA_HOST + BARS_PATH, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        calls += 1
        first = first or {h: resp.headers.get(h) for h in LIMIT_HEADERS}
        body = resp.json()
        bars += (body.get("bars") or {}).get(SYMBOL, [])
        if not body.get("next_page_token"):
            break
        params["page_token"] = body["next_page_token"]
    df = pd.DataFrame(bars)
    if not df.empty:
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df, perf_counter() - t0, calls, first


def describir(df, timeframe, secs, calls, first):
    ny = df["timestamp"].dt.tz_convert("America/New_York")
    horas = sorted({f"{t.hour:02d}:{t.minute:02d}" for t in ny})
    print(f"[{timeframe}] velas={len(df)} primera={df['timestamp'].iloc[0]} pedidos={calls} "
          f"tiempo={secs:.2f}s horas NY (UTC en 1d)={horas[:12]}{'...' if len(horas) > 12 else ''}")
    print(f"[{timeframe}] headers de límite: {first}")


def senales(candles):
    out = {}
    for name in ("sma-cross", "momentum-12m"):
        instance = build_strategy(cli_aliases()[name])
        out[name] = bar_signals(instance, candles, SYMBOL, "1d")["entry"]
    return pd.DataFrame(out)


def main():
    source = AlpacaSource()
    try:
        headers = source._headers()
    except RuntimeError as exc:
        sys.exit(f"sin claves: {exc}")
    for feed in ("sip", "iex"):
        try:
            df, *_ = pedir(source, headers, "1d", feed=feed, start=pd.Timestamp("2024-01-02", tz="UTC"))
            print(f"feed={feed}: responde, {len(df)} velas diarias desde 2024")
        except requests.HTTPError as exc:
            print(f"feed={feed}: rechazado ({exc.response.status_code}: {exc.response.text[:120]})")
    for timeframe in ("1h", "4h", "1d"):
        df, secs, calls, first = pedir(source, headers, timeframe)
        describir(df, timeframe, secs, calls, first)
    split, *_ = pedir(source, headers, "1d", adjustment="split")
    todo, *_ = pedir(source, headers, "1d", adjustment="all")
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    a = split[cols].set_index("timestamp", drop=False)
    b = todo[cols].set_index("timestamp", drop=False)
    comunes = a.index.intersection(b.index)
    dif = (a.loc[comunes, "close"] / b.loc[comunes, "close"] - 1).abs()
    print(f"split vs all (1d): velas {len(a)}/{len(b)}, cierre difiere hasta {dif.max():.4%}, "
          f"mediana {dif.median():.4%}")
    sa, sb = senales(a.loc[comunes]), senales(b.loc[comunes])
    for name in sa:
        cambia = int((sa[name] != sb[name]).sum())
        print(f"{name}: entradas split={int((sa[name] != 0).sum())} all={int((sb[name] != 0).sum())} "
              f"velas con señal distinta={cambia}")


if __name__ == "__main__":
    main()
