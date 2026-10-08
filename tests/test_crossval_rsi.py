"""QuantAgent-832.1: el port RSI a backtesting.py (scripts/crossval_rsi.py) corre y es consistente.

Se omite si `backtesting` no esta instalado (es dependencia de desarrollo; CI instala solo `pip install -e .`).
No compara contra el motor del proyecto: valida el formato, la alineacion con el fixture y el determinismo.
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("backtesting")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "crossval_rsi.py"
FIXTURE = ROOT / "tests" / "fixtures" / "spy-90d.csv"
EXPECTED_COLUMNS = [
    "entry_time", "exit_time", "symbol", "side", "qty",
    "entry_price", "exit_price", "stop_loss", "pnl", "exit_reason",
]
EXPECTED_PARAMS = [
    "rsi_period", "oversold_threshold", "overbought_threshold", "stop_loss_pct", "take_profit_pct",
    "trailing_stop_pct", "initial_cash", "base_position_pct", "slippage_pct", "commission",
]


def _run(out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )


@pytest.fixture(scope="module")
def run_twice(tmp_path_factory):
    d = tmp_path_factory.mktemp("crossval")
    return _run(d / "a.csv"), _run(d / "b.csv"), d / "a.csv", d / "b.csv"


def _fixture_closes() -> dict:
    with open(FIXTURE, newline="") as f:
        return {r["timestamp"]: float(r["close"]) for r in csv.DictReader(f)}


def test_script_exits_zero_and_writes_columns(run_twice):
    proc, _, out, _ = run_twice
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(out, newline="") as f:
        reader = csv.reader(f)
        assert next(reader) == EXPECTED_COLUMNS
        assert sum(1 for _ in reader) >= 1


def test_param_table_has_equal_values(run_twice):
    rows = {}
    for line in run_twice[0].stdout.splitlines():
        m = re.fullmatch(r"(\w+)\s+(\S+)\s+(\S+)\s+(SI|NO)", line.strip())
        if m:
            rows[m.group(1)] = m.groups()[1:]
    assert list(rows) == EXPECTED_PARAMS
    for name, (proj, port, same) in rows.items():
        assert same == "SI" and float(proj) == float(port), name


def test_trades_are_aligned_with_fixture_candles(run_twice):
    """Fill at the close of the signal candle: times are fixture timestamps and prices are those candles' closes."""
    closes = _fixture_closes()
    with open(run_twice[2], newline="") as f:
        trades = list(csv.DictReader(f))
    assert trades
    for t in trades:
        assert t["entry_time"] in closes and t["exit_time"] in closes
        assert t["exit_time"] >= t["entry_time"]
        assert float(t["entry_price"]) == closes[t["entry_time"]]
        assert float(t["exit_price"]) == closes[t["exit_time"]]
        assert t["side"] in ("buy", "sell") and float(t["qty"]) > 0
        sign = 1 if t["side"] == "buy" else -1
        expected = sign * float(t["qty"]) * (float(t["exit_price"]) - float(t["entry_price"]))
        assert float(t["pnl"]) == pytest.approx(expected, rel=1e-9, abs=1e-6)
        assert t["stop_loss"] == "" and t["exit_reason"] == ""  # sin equivalente en backtesting.py


def test_output_is_byte_identical_across_runs(run_twice):
    assert run_twice[1].returncode == 0
    assert run_twice[2].read_bytes() == run_twice[3].read_bytes()
