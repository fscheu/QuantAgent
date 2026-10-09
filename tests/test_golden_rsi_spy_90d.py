"""Golden regression test for deterministic RSI strategy on spy-90d fixture (QuantAgent-piv, QuantAgent-al2.3).

Freezes trades and audited metrics from M1 (with intrabar stops enabled by default):
- 340 closed trades matching tests/fixtures/golden/rsi-spy-90d-trades.csv row-by-row
- Reference metrics: Win rate 33.24%, Profit factor 1.35, Sharpe ratio 9.46,
  Total PnL 4571.27, Slippage 0.05% por lado, max drawdown 0.001414
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine

from quantagent.models import Base

GOLDEN_CSV_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "golden" / "rsi-spy-90d-trades.csv"
)

EXPECTED_METRICS = {
    "Trades": "340",
    "Win rate": "33.24%",
    "Profit factor": "1.35",
    "Sharpe ratio": "9.46",
    "Total PnL": "4571.27",
    "Slippage": "0.05% por lado",
    "max drawdown": "0.001414",
}


def test_golden_rsi_spy_90d_metrics_and_trades(tmp_path):
    """Runs rsi/spy-90d via CLI on a clean DB and verifies exact metrics and trade log."""
    assert GOLDEN_CSV_PATH.exists(), f"Golden CSV not found at {GOLDEN_CSV_PATH}"

    db_url = f"sqlite:///{tmp_path / 'golden.db'}"
    Base.metadata.create_all(create_engine(db_url))

    out_trades = tmp_path / "trades.csv"
    out_equity = tmp_path / "equity.csv"
    cmd = [
        sys.executable, "-m", "quantagent.cli", "backtest", "run",
        "--strategy", "rsi", "--fixture", "spy-90d",
        "--out", str(out_trades), "--equity-out", str(out_equity),
    ]
    env = {**os.environ, "DATABASE_URL": db_url, "OPENAI_API_KEY": ""}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    assert proc.returncode == 0, f"CLI run failed (exit {proc.returncode}):\n{proc.stdout}"

    # 1. Parse and assert metrics from stdout
    metrics: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            metrics[key.strip()] = val.strip()
        if "(max drawdown:" in line:
            m = re.search(r"\(max drawdown:\s*([0-9.]+)\)", line)
            if m:
                metrics["max drawdown"] = m.group(1)

    for metric_name, expected_val in EXPECTED_METRICS.items():
        actual_val = metrics.get(metric_name)
        assert actual_val == expected_val, (
            f"Metric mismatch '{metric_name}': actual '{actual_val}' != expected '{expected_val}'"
        )

    # 2. Compare trade log row-by-row with golden fixture
    assert out_trades.exists(), "Output trades CSV was not generated"
    with GOLDEN_CSV_PATH.open(newline="", encoding="utf-8") as f_gold:
        golden_rows = list(csv.reader(f_gold))
    with out_trades.open(newline="", encoding="utf-8") as f_actual:
        actual_rows = list(csv.reader(f_actual))

    assert len(actual_rows) == len(golden_rows), (
        f"CSV row count mismatch: actual has {len(actual_rows)}, golden has {len(golden_rows)}"
    )

    for idx, (actual_row, golden_row) in enumerate(zip(actual_rows, golden_rows)):
        assert actual_row == golden_row, (
            f"Trade mismatch at row {idx}: actual={actual_row} != golden={golden_row}"
        )
