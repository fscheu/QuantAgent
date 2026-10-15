import inspect
import math
import statistics

import pandas as pd
import pytest

from quantagent.backtesting.vbt_filter import equity_metrics, run_filter

COMMISSION, SLIPPAGE, CAPITAL = 0.001, 0.0005, 10_000.0
OPEN = [100, 101, 102, 104, 103, 105, 110, 108, 107, 106]


def _candles():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame({"open": [float(x) for x in OPEN], "close": [float(x) + 0.5 for x in OPEN]}, index=idx)

def _signals(candles, entry=None, exits=()):
    s = pd.DataFrame({"entry": 0, "exit": False}, index=candles.index)
    for i, v in (entry or {}).items():
        s.iloc[i, s.columns.get_loc("entry")] = v
    for i in exits:
        s.iloc[i, s.columns.get_loc("exit")] = True
    return s

def _run(signals, candles):
    return run_filter(signals, candles, commission_pct=COMMISSION, slippage_pct=SLIPPAGE, capital=CAPITAL)


def test_long_fills_at_next_open_with_costs():
    c = _candles()
    r = _run(_signals(c, {2: 1}, exits=[6]), c)
    buy = OPEN[3] * (1 + SLIPPAGE)  # señal en vela 2 -> open de la 3
    sell = OPEN[7] * (1 - SLIPPAGE)  # salida en vela 6 -> open de la 7
    qty = CAPITAL / (buy * (1 + COMMISSION))
    expected = qty * sell * (1 - COMMISSION) - CAPITAL
    assert r["trades"] == 1
    assert r["pnl"] == pytest.approx(expected, abs=1e-9)


def test_short_fills_at_next_open_with_costs():
    c = _candles()
    r = _run(_signals(c, {2: -1}, exits=[6]), c)
    sell = OPEN[3] * (1 - SLIPPAGE)
    cover = OPEN[7] * (1 + SLIPPAGE)
    qty = CAPITAL / (sell * (1 + COMMISSION))
    expected = qty * sell * (1 - COMMISSION) - qty * cover * (1 + COMMISSION)
    assert r["pnl"] == pytest.approx(expected, rel=1e-6)


def test_opposite_entry_reverses():
    c = _candles()
    r = _run(_signals(c, {2: 1, 5: -1}, exits=[8]), c)
    assert r["trades"] == 2


def test_signal_on_last_candle_opens_nothing():
    c = _candles()
    r = _run(_signals(c, {9: 1}), c)
    assert r["trades"] == 0
    assert r["pnl"] == 0
    assert r["max_drawdown"] == 0


def test_no_close_fill_option():
    params = inspect.signature(run_filter).parameters
    assert set(params) == {"signals", "candles", "commission_pct", "slippage_pct", "capital"}


def test_sharpe_and_drawdown_by_hand_on_small_curve():
    eq = pd.Series([100.0, 110.0, 99.0, 108.9], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05"]))
    ppy = 3 / (4 * 86400 / (365.25 * 86400))  # 3 retornos en 4 días de calendario
    rets = [0.1, -0.1, 0.1]
    mean, sd = statistics.mean(rets), statistics.stdev(rets)
    m, m0 = equity_metrics(eq), equity_metrics(eq, 0.0)
    assert m["max_drawdown"] == pytest.approx((110 - 99) / 110, abs=1e-12)
    assert m["sharpe"] == pytest.approx((mean - 0.02 / ppy) / sd * math.sqrt(ppy), abs=1e-9)
    assert m0["sharpe"] == pytest.approx(mean / sd * math.sqrt(ppy), abs=1e-9)
