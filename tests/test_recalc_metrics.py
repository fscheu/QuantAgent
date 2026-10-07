"""Tests for scripts/recalc_metrics.py (QuantAgent-hx0.5), the engine-independent metrics recalculation.

The expected values are worked out by hand in the comments, never taken from the engine: a
regression here means the "second opinion" used to audit the backtest metrics is itself wrong.
"""

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


def test_profit_factor_without_losing_trades_is_inf(tmp_path):
    """Validates the division-by-zero edge: with wins and no losses the profit factor is infinite."""
    only_winners = HEADER + "".join(FIVE_TRADES.splitlines(keepends=True)[i] for i in (1, 3))
    assert _run(tmp_path, only_winners) == [
        "Trades: 2",
        "Win rate: 100.00%",
        "Profit factor: inf",
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


def test_script_does_not_import_the_engine():
    """Validates the independence claim: loading the script must not pull in `quantagent`."""
    code = f"import runpy, sys; runpy.run_path({str(SCRIPT)!r}); sys.exit('quantagent' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# Synthetic curve (100 -> 120 -> 90 -> 130 -> 100 -> 140) yields peak 120 -> trough 90 = 25% max DD.
SIX_POINT_CURVE = "equity\n100\n120\n90\n130\n100\n140\n"

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
        "Sharpe ratio: 13.61",
        "Total PnL: 60.00",
        "PnL por trade: 5/5 filas coinciden",
        "Max drawdown: 0.250000",
    ]
