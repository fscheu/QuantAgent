"""QuantAgent-a03: control con backtesting.py sobre SPY diario real (snapshot etf-1d-2026-10)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.snapshot, pytest.mark.oraculo]

pytest.importorskip("backtesting")

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = os.environ.get("QUANTAGENT_SNAPSHOT_DIR")
if not SNAPSHOT_DIR or not (Path(SNAPSHOT_DIR) / "etf-1d-2026-10").exists():
    pytest.skip("Snapshot etf-1d-2026-10 no disponible", allow_module_level=True)


def test_crossval_snapshot_reports_comparison_and_double_touches():
    cmd = [
        sys.executable,
        "scripts/crossval_compare.py",
        "--snapshot",
        "etf-1d-2026-10",
        "--symbol",
        "SPY",
        "--from",
        "2007-01-03",
        "--to",
        "2018-12-31",
        "--intrabar",
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert "Trades: motor=180 port=187" in p.stdout
    assert "Velas que tocan stop y take profit: 3" in p.stdout
    assert "PRIMER TRADE DIFERENTE: nº 2" in p.stdout


def test_crossval_rsi_snapshot_runs(tmp_path):
    out = tmp_path / "port.csv"
    cmd = [
        sys.executable,
        "scripts/crossval_rsi.py",
        "--snapshot",
        "etf-1d-2026-10",
        "--symbol",
        "SPY",
        "--from",
        "2007-01-03",
        "--to",
        "2018-12-31",
        "--out",
        str(out),
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert "Trades: 153" in p.stdout
    assert out.exists()
