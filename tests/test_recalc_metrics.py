"""Tests for scripts/recalc_metrics.py (QuantAgent-hx0.5), the engine-independent metrics recalculation.

The expected values are worked out by hand in the comments, never taken from the engine: a
regression here means the "second opinion" used to audit the backtest metrics is itself wrong.
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recalc_metrics.py"
HEADER = "entry_time,exit_time,symbol,side,qty,entry_price,exit_price,stop_loss,pnl,exit_reason\n"

# 5 closed trades with pnl +100, -50, +30, -20 and 0 (breakeven), plus one still-open row.
FIVE_TRADES = HEADER + (
    "2026-01-02T05:00:00,2026-01-02T13:00:00,SPY,buy,10,100,110,95,100,TAKE_PROFIT\n"
    "2026-01-02T13:00:00,2026-01-02T17:00:00,SPY,sell,10,110,115,115,-50,STOP_LOSS\n"
    "2026-01-03T05:00:00,2026-01-03T09:00:00,SPY,buy,10,100,103,95,30,SIGNAL\n"
    "2026-01-03T09:00:00,2026-01-03T13:00:00,SPY,sell,10,103,105,108,-20,SIGNAL\n"
    "2026-01-04T05:00:00,2026-01-04T09:00:00,SPY,buy,10,100,100,95,0,SIGNAL\n"
    "2026-01-05T05:00:00,,SPY,buy,10,100,,95,,\n"
)


def _run(tmp_path, content):
    csv_path = tmp_path / "run.csv"
    csv_path.write_text(content)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(csv_path)], capture_output=True, text=True, check=True
    ).stdout.splitlines()


def test_five_handmade_trades_give_the_hand_computed_metrics(tmp_path):
    """Validates the 4 formulas on a case with winners, losers, a breakeven and an open row."""
    assert _run(tmp_path, FIVE_TRADES) == [
        "Trades: 5",  # the open row (empty pnl) is not counted
        "Win rate: 40.00%",  # 2 winners / 5 trades; the breakeven is not a win
        "Profit factor: 1.86",  # (100 + 30) / (50 + 20) = 1.857...
        "Total PnL: 60.00",  # 100 - 50 + 30 - 20 + 0
        "PnL por trade: 5/5 filas coinciden",
    ]


def test_profit_factor_without_losing_trades_is_na(tmp_path):
    """Validates the division-by-zero edge: with wins and no losses the profit factor is printed as n/a."""
    only_winners = HEADER + "".join(FIVE_TRADES.splitlines(keepends=True)[i] for i in (1, 3))
    assert _run(tmp_path, only_winners) == [
        "Trades: 2",
        "Win rate: 100.00%",
        "Profit factor: n/a",
        "Total PnL: 130.00",
        "PnL por trade: 2/2 filas coinciden",
    ]


def test_mismatched_pnl_exits_with_code_1(tmp_path):
    """Validates that a row whose pnl != (exit - entry) * qty triggers exit 1."""
    bad_csv = HEADER + "2026-01-02T05:00:00,2026-01-02T13:00:00,SPY,buy,10,100,110,95,999,TAKE_PROFIT\n"
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(bad_csv)
    proc = subprocess.run([sys.executable, str(SCRIPT), str(csv_path)], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "PnL por trade: 0/1 filas coinciden" in proc.stdout


# Comisión 0.1% por lado (QuantAgent-40o). Long 10 @ 100 -> 110: bruto 100, comisión 0.001 * 10 * (100 + 110) = 2.10.
# Short 10 @ 110 -> 105: bruto 50, comisión 0.001 * 10 * (110 + 105) = 2.15.
COMMISSION_TRADES = HEADER + (
    "2026-01-02T05:00:00,2026-01-02T13:00:00,SPY,buy,10,100,110,95,97.90,TAKE_PROFIT\n"
    "2026-01-02T13:00:00,2026-01-02T17:00:00,SPY,sell,10,110,105,115,47.85,SIGNAL\n"
)


def test_commission_pct_discounts_entry_and_exit_commission(tmp_path):
    """Validates the per-trade formula with commission: pnl = bruto - pct * qty * (entry + exit)."""
    csv_path = tmp_path / "run.csv"
    csv_path.write_text(COMMISSION_TRADES)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(csv_path), "--commission-pct", "0.001"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "Total PnL: 145.75" in out  # 97.90 + 47.85
    assert "PnL por trade: 2/2 filas coinciden" in out


def test_commission_pct_flags_a_pnl_missing_the_entry_commission(tmp_path):
    """Validates the mismatch report: a pnl net of only the exit commission (100 - 1.10) is off by the entry one (1.00)."""
    csv_path = tmp_path / "run.csv"
    csv_path.write_text(HEADER + "2026-01-02T05:00:00,2026-01-02T13:00:00,SPY,buy,10,100,110,95,98.90,SIGNAL\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(csv_path), "--commission-pct", "0.001"], capture_output=True, text=True
    )
    assert proc.returncode == 1
    assert "PnL por trade: 0/1 filas coinciden" in proc.stdout
    assert "Diferencia CSV - recalculado: total 1.00, min 1.00, max 1.00" in proc.stdout


def test_script_does_not_import_the_engine():
    """Validates the independence claim: loading the script must not pull in `quantagent`."""
    code = f"import runpy, sys; runpy.run_path({str(SCRIPT)!r}); sys.exit('quantagent' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# Synthetic curve (100 -> 120 -> 90 -> 130 -> 100 -> 140) yields peak 120 -> trough 90 = 25% max DD.
SIX_POINT_CURVE = (
    "timestamp,equity\n"
    "2026-01-01T00:00:00,100\n"
    "2026-01-01T01:00:00,120\n"
    "2026-01-01T02:00:00,90\n"
    "2026-01-01T03:00:00,130\n"
    "2026-01-01T04:00:00,100\n"
    "2026-01-01T05:00:00,140\n"
)

# Returns of 1%, 2%, 3% (mean 2%, std 1%) with rf=2% and N=4 yield (0.02 - 0.005)/0.01 * 2 = Sharpe 3.00.
KNOWN_SHARPE_CURVE = "equity\n1000\n1010\n1030.2\n1061.106\n"


def test_synthetic_curve_max_drawdown_is_25_percent(tmp_path):
    """Synthetic 6-point curve (100, 120, 90, 130, 100, 140) yields exactly 25% max drawdown."""
    eq_path = tmp_path / "eq.csv"
    eq_path.write_text(SIX_POINT_CURVE)
    lines = subprocess.run(
        [sys.executable, str(SCRIPT), "--equity", str(eq_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "Max drawdown: 0.250000" in lines


def test_known_sharpe_series_gives_hand_computed_sharpe(tmp_path):
    """Returns of 1%, 2%, 3% (mean 2%, std 1%) with rf=2% annualized and N=4 yield Sharpe 3.00."""
    eq_path = tmp_path / "eq.csv"
    eq_path.write_text(KNOWN_SHARPE_CURVE)
    lines = subprocess.run(
        [sys.executable, str(SCRIPT), "--equity", str(eq_path), "--periods-per-year", "4"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "Sharpe ratio: 3.00" in lines
    assert "Max drawdown: 0.000000" in lines


def test_recalc_with_both_trades_and_equity_prints_all_metrics(tmp_path):
    """Combining trades and equity outputs trade metrics and equity metrics."""
    trades_path = tmp_path / "run.csv"
    trades_path.write_text(FIVE_TRADES)
    eq_path = tmp_path / "eq.csv"
    eq_path.write_text(SIX_POINT_CURVE)
    lines = subprocess.run(
        [sys.executable, str(SCRIPT), str(trades_path), "--equity", str(eq_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert lines == [
        "Trades: 5",
        "Win rate: 40.00%",
        "Profit factor: 1.86",
        "Sharpe ratio: 31.48",
        "Total PnL: 60.00",
        "PnL por trade: 5/5 filas coinciden",
        "Max drawdown: 0.250000",
    ]


def test_synthetic_equity_same_sharpe_in_both_calendars_without_calendar_param(tmp_path):
    """Equity with known Sharpe in two calendars (24/7 and market hours) gives expected Sharpe."""
    import datetime
    import math

    def make_csv(n_periods, target_sharpe=2.0, target_vol=0.20, rf=0.02):
        target_mean = (target_sharpe * target_vol + rf) / n_periods
        target_std = target_vol / math.sqrt(n_periods)
        d = target_std * math.sqrt((n_periods - 1) / n_periods)
        rets = [target_mean + d if i % 2 == 0 else target_mean - d for i in range(n_periods)]

        t0 = datetime.datetime(2026, 1, 1, 0, 0, 0)
        t1 = datetime.datetime(2027, 1, 1, 6, 0, 0)  # 365.25 days
        dt = (t1 - t0) / n_periods

        lines = ["timestamp,equity\n", f"{t0.isoformat()},100000.0\n"]
        curr = t0
        eq = 100000.0
        for r in rets:
            curr += dt
            eq *= (1.0 + r)
            lines.append(f"{curr.isoformat()},{eq:.6f}\n")
        return "".join(lines)

    cal_24_7 = tmp_path / "cal_24_7.csv"
    cal_24_7.write_text(make_csv(8766))

    cal_mkt = tmp_path / "cal_mkt.csv"
    cal_mkt.write_text(make_csv(1638))

    # Both yield Sharpe 2.00 without any calendar parameter passed
    out_24_7 = subprocess.run(
        [sys.executable, str(SCRIPT), "--equity", str(cal_24_7)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "Sharpe ratio: 2.00" in out_24_7

    out_mkt = subprocess.run(
        [sys.executable, str(SCRIPT), "--equity", str(cal_mkt)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "Sharpe ratio: 2.00" in out_mkt

    # Fixed table (N=1638 for 1h) fails for 24/7 (yields 0.68 instead of 2.00)
    out_fixed = subprocess.run(
        [sys.executable, str(SCRIPT), "--equity", str(cal_24_7), "--periods-per-year", "1638"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert "Sharpe ratio: 0.68" in out_fixed


# Comprar y mantener (QuantAgent-bzt): 3 velas diarias 100 -> 90 -> 120, capital 10000.
THREE_CANDLES = (
    "symbol,timeframe,timestamp,open,high,low,close,volume\n"
    "SPY,1d,2024-01-01T00:00:00,100,100,100,100,1\n"
    "SPY,1d,2024-01-02T00:00:00,90,90,90,90,1\n"
    "SPY,1d,2024-01-03T00:00:00,120,120,120,120,1\n"
)


def _buy_and_hold(path, *extra):
    cmd = [sys.executable, str(SCRIPT), "--comprar-y-mantener", str(path), "--capital", "10000", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def test_buy_and_hold_three_candles_with_costs_gives_the_hand_computed_line(tmp_path):
    """Validates PnL, Sharpe and max drawdown of buy and hold with slippage 0.1% and commission 0.2% per side.

    qty = 10000 / (100 * 1.001 * 1.002) = 99.70070; vende a 120 * 0.999 menos 0.2%: PnL = 1928.22.
    Equity qty*100, qty*90, venta: retornos -0.1 y 120 * 0.999 * 0.998 / 90 - 1 = 0.329336;
    N = 2 / (2 / 365.25) = 365.25 -> Sharpe 7.22. El pozo 100 -> 90 es 10%.
    """
    csv_path = tmp_path / "velas.csv"
    csv_path.write_text(THREE_CANDLES)
    out = _buy_and_hold(csv_path, "--slippage-pct", "0.001", "--commission-pct", "0.002")
    assert out == "Comprar y mantener: PnL 1928.22, Sharpe 7.22, max drawdown 0.100000"


def test_buy_and_hold_reads_the_adjusted_close_from_a_snapshot_parquet(tmp_path):
    """Validates the parquet path: uses adj_close (100 -> 95 -> 130), not close: PnL 10000 * 0.30, pozo 5%."""
    import pandas as pd

    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    path = tmp_path / "SPY.parquet"
    pd.DataFrame({"close": [100.0, 90.0, 120.0], "adj_close": [100.0, 95.0, 130.0]}, index=idx).to_parquet(path)
    assert _buy_and_hold(path).startswith("Comprar y mantener: PnL 3000.00, Sharpe ")
    assert _buy_and_hold(path).endswith(", max drawdown 0.050000")


def test_buy_and_hold_matches_the_cli_line_on_spy_smoke(tmp_path):
    """Validates the cross-check: the recalculation equals the engine's 'Comprar y mantener' line on spy-smoke."""
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'bh.db'}", "OPENAI_API_KEY": ""}
    subprocess.run([sys.executable, "-c", "from quantagent.database import init_db; init_db()"], env=env, check=True)
    cli = subprocess.run(
        [sys.executable, "-m", "quantagent.cli", "backtest", "run", "--strategy", "rsi", "--fixture", "spy-smoke",
         "--commission-pct", "0.001"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout
    engine_line = next(line for line in cli.splitlines() if line.startswith("Comprar y mantener"))
    fixture = Path(__file__).resolve().parent / "fixtures" / "spy-smoke.csv"
    recalc_line = subprocess.run(
        [sys.executable, str(SCRIPT), "--comprar-y-mantener", str(fixture), "--slippage-pct", "0.0005",
         "--commission-pct", "0.001"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert recalc_line == engine_line
