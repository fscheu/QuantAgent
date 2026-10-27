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
    o = [float(x) for x in OPEN]
    return pd.DataFrame({"open": o, "high": [x + 1 for x in o], "low": [x - 1 for x in o], "close": [x + 0.5 for x in o]}, index=idx)

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


def _ohlc(rows):
    idx = pd.date_range("2020-01-01", periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).astype(float)

def _stop_run(rows, side=1, **pcts):
    c = _ohlc(rows)
    s = _signals(c, {0: side})
    for k, v in pcts.items():
        s[k] = float("nan")
        s.iloc[0, s.columns.get_loc(k)] = v
    r = run_filter(s, c, commission_pct=0.0, slippage_pct=0.0, capital=CAPITAL)
    return r["pnl"], CAPITAL / c["open"].iloc[1]  # (pnl, cantidad) con entrada al open de la vela 1

def test_stop_loss_exits_at_stop_level_on_crossing_candle():
    pnl, qty = _stop_run([(100, 100, 100, 100), (100, 101, 99, 100), (100, 100, 90, 92), (92, 93, 80, 85)], stop_loss_pct=0.05)
    assert pnl == pytest.approx(qty * (95 - 100))  # sale a 95 en la vela 2, no al close ni al low

def test_candle_touching_stop_and_take_profit_exits_by_stop():
    pnl, qty = _stop_run([(100, 100, 100, 100), (100, 101, 99, 100), (100, 112, 94, 110), (110, 111, 109, 110)],
                         stop_loss_pct=0.05, take_profit_pct=0.10)
    assert pnl == pytest.approx(qty * (95 - 100))

def test_gap_below_stop_exits_at_open():
    pnl, qty = _stop_run([(100, 100, 100, 100), (100, 101, 99, 100), (90, 91, 88, 89), (89, 90, 85, 86)], stop_loss_pct=0.05)
    assert pnl == pytest.approx(qty * (90 - 100))


def _trail_candles(low_dip_bar=None):
    closes = [100.0, 100.0, 110.0, 120.0, 112.0, 111.0, 109.0, 108.0]
    opens = [closes[0]] + closes[:-1]
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    c = pd.DataFrame({"open": opens, "close": closes}, index=idx)
    c["high"] = c[["open", "close"]].max(axis=1) + 0.5
    c["low"] = c[["open", "close"]].min(axis=1) - 0.5
    if low_dip_bar is not None:
        c.iloc[low_dip_bar, c.columns.get_loc("low")] = 97.0  # toca el stop fijo de 2%
    return c


def _trail_signals(c, side=1):
    s = _signals(c, {0: side})
    s["stop_loss_pct"], s["trailing_stop_pct"] = 0.02, 0.05
    return s


def test_trailing_exits_next_open_after_close_below_max_close():
    c = _trail_candles()
    r = _run(_trail_signals(c), c)
    # máximo cierre 120 (vela 3); el cierre 112 de la vela 4 < 120*0.95 -> sale en el open de la 5 (=112)
    buy = c["open"].iloc[1] * (1 + SLIPPAGE)
    sell = c["open"].iloc[5] * (1 - SLIPPAGE)
    qty = CAPITAL / (buy * (1 + COMMISSION))
    assert r["trades"] == 1
    assert r["pnl"] == pytest.approx(qty * sell * (1 - COMMISSION) - CAPITAL, abs=1e-6)


def test_fixed_stop_inside_candle_wins_over_trailing():
    c = _trail_candles(low_dip_bar=2)
    r = _run(_trail_signals(c), c)
    buy = c["open"].iloc[1] * (1 + SLIPPAGE)
    assert r["trades"] == 1
    assert r["pnl"] < 0  # salió por el stop 2% en la vela 2, antes del trailing
    qty = CAPITAL / (buy * (1 + COMMISSION))
    assert r["pnl"] == pytest.approx(qty * buy * 0.98 * (1 - SLIPPAGE) * (1 - COMMISSION) - CAPITAL, abs=0.01)



def test_trailing_mark_of_later_trade_does_not_close_reentered_trade():
    # A sale en la 4 (5%); la señal de la 5 abre A2 (20%). La marca de B (señal 9, 5%) cae en la 13: no debe cerrar a A2.
    closes = [100.0, 100, 110, 120, 112, 112, 112, 112, 112, 112, 112, 130, 140, 130, 130, 150, 150]
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    c = pd.DataFrame({"open": [closes[0]] + closes[:-1], "close": closes}, index=idx)
    c["high"] = c[["open", "close"]].max(axis=1) + 0.5
    c["low"] = c[["open", "close"]].min(axis=1) - 0.5
    c.iloc[7, c.columns.get_loc("low")] = 97.0  # stop fijo de A en la 7 (A2 no tiene stop)
    s = _signals(c, {0: 1, 5: 1, 9: 1})
    s["stop_loss_pct"] = [0.02] + [float("nan")] * 16
    s["trailing_stop_pct"] = [0.05] * 5 + [0.20] + [0.05] * 11
    r = _run(s, c)
    o = c["open"]
    cash = CAPITAL / (o.iloc[1] * (1 + SLIPPAGE) * (1 + COMMISSION)) * o.iloc[5] * (1 - SLIPPAGE) * (1 - COMMISSION)
    equity = cash / (o.iloc[6] * (1 + SLIPPAGE) * (1 + COMMISSION)) * closes[-1]  # A2 abierta hasta el final
    assert r["trades"] == 2
    assert r["pnl"] == pytest.approx(equity - CAPITAL, abs=0.01)


@pytest.mark.parametrize("rows, exits, expiry, expected_trades, expected_pnl", [
    ([(90, 92, 89, 91), (95, 103, 94, 98), (98, 106, 97, 105), (105, 107, 104, 106)], [2], 3, 1, CAPITAL / 100 * 5),
    ([(90, 92, 89, 91), (105, 110, 103, 108), (108, 112, 107, 110), (108, 110, 106, 107)], [2], 3, 1, CAPITAL / 105 * 3),
    ([(90, 92, 89, 91), (91, 95, 90, 93), (92, 96, 91, 94), (105, 110, 104, 108)], [], 2, 0, 0.0),
])
def test_stop_entry_cases(rows, exits, expiry, expected_trades, expected_pnl):
    c = _ohlc(rows)
    s = _signals(c, {0: 1}, exits=exits)
    s["entry_order"] = ["stop"] + ["market"] * (len(c) - 1)
    s["entry_price"] = [100.0] + [float("nan")] * (len(c) - 1)
    s["entry_expiry_bars"] = [expiry] + [float("nan")] * (len(c) - 1)
    r = run_filter(s, c, commission_pct=0.0, slippage_pct=0.0, capital=CAPITAL)
    assert r["trades"] == expected_trades
    assert r["pnl"] == pytest.approx(expected_pnl)


def test_chandelier_exit_long_and_short_synthetic():
    # Long: sube y cae; dispara en vela 4 (close 108 <= max(high) 115 - 6 = 109), sale al open de vela 5 (107).
    rows_long = [
        (100, 100, 100, 100),  # 0: señal
        (100, 105, 99, 104),   # 1: entra a open 100. max_high=105, stop=100, close=104 > 100
        (104, 110, 103, 109),  # 2: max_high=110, stop=105, close=109 > 105
        (109, 115, 108, 114),  # 3: max_high=115, stop=109, close=114 > 109
        (114, 114, 107, 108),  # 4: max_high=115, stop=109, close=108 <= 109 -> dispara salida
        (107, 108, 106, 107),  # 5: llena salida al open 107
        (107, 107, 100, 101),  # 6
    ]
    c_long = _ohlc(rows_long)
    s_long = _signals(c_long, {0: 1})
    s_long["chandelier_dist"] = [5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]
    r_long = run_filter(s_long, c_long, commission_pct=0.0, slippage_pct=0.0, capital=CAPITAL)
    assert r_long["trades"] == 1
    recs_long = r_long["portfolio"].trades.records
    assert int(recs_long["entry_idx"].iloc[0]) == 1
    assert int(recs_long["exit_idx"].iloc[0]) == 5
    assert float(recs_long["exit_price"].iloc[0]) == 107.0
    assert r_long["pnl"] == pytest.approx(CAPITAL / 100.0 * (107.0 - 100.0))

    # Corto: cae y sube; dispara en vela 3 (close 96 >= min(low) 90 + 5 = 95), sale al open de vela 4 (97).
    rows_short = [
        (100, 100, 100, 100),  # 0: señal
        (100, 101, 95, 96),    # 1: entra a open 100. min_low=95, stop=100, close=96 < 100
        (96, 97, 90, 91),      # 2: min_low=90, stop=95, close=91 < 95
        (91, 98, 90, 96),      # 3: min_low=90, stop=95, close=96 >= 95 -> dispara salida
        (97, 99, 95, 98),      # 4: llena salida al open 97
        (98, 98, 94, 95),      # 5
    ]
    c_short = _ohlc(rows_short)
    s_short = _signals(c_short, {0: -1})
    s_short["chandelier_dist"] = 5.0
    r_short = run_filter(s_short, c_short, commission_pct=0.0, slippage_pct=0.0, capital=CAPITAL)
    assert r_short["trades"] == 1
    recs_short = r_short["portfolio"].trades.records
    assert int(recs_short["entry_idx"].iloc[0]) == 1
    assert int(recs_short["exit_idx"].iloc[0]) == 4
    assert float(recs_short["exit_price"].iloc[0]) == 97.0
    assert r_short["pnl"] == pytest.approx(CAPITAL / 100.0 * (100.0 - 97.0))

