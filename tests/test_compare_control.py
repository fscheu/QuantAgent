"""Tests para scripts/compare_control.py (V23b, QuantAgent-k6n9)."""

import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("backtesting")
from click.testing import CliRunner
from sqlalchemy import create_engine

import quantagent.database as database
from quantagent import settings
from quantagent.backtesting.fixtures import FIXTURES_DIR
from quantagent.cli.backtest import backtest_group
from quantagent.data.snapshot import write_snapshot
from quantagent.models import Base

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compare_control.py"


def _daily_frame() -> pd.DataFrame:
    with (FIXTURES_DIR / "spy-2y-1d.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    df = pd.DataFrame(
        {
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [int(r["volume"]) for r in rows],
        },
        index=pd.DatetimeIndex([r["timestamp"] for r in rows], name="timestamp"),
    )
    df.insert(4, "adj_close", df["close"])
    return df


@pytest.fixture
def snap_env(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    monkeypatch.setenv("QUANTAGENT_RESERVA_DESDE", "2030-01-01")
    monkeypatch.setenv("QUANTAGENT_PRUEBA_DESDE", "2030-01-01")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    original_database_url = settings.DATABASE_URL
    try:
        yield CliRunner(), tmp_path
    finally:
        settings.DATABASE_URL = original_database_url
        database._engine = None
        database._SessionLocal = None


def _run_control(tabla: Path, trades_dir: Path, snapshot: str, md: Path | None = None) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--tabla", str(tabla),
        "--trades-dir", str(trades_dir),
        "--snapshot", snapshot,
    ]
    if md:
        cmd.extend(["--md", str(md)])
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_compare_control_coincide_y_falla_si_se_altera_un_trade(snap_env):
    runner, tmp_path = snap_env
    write_snapshot("t", {"SPY": _daily_frame()})
    out_csv = tmp_path / "cmp.csv"
    trades_dir = tmp_path / "trades"

    res = runner.invoke(
        backtest_group,
        [
            "compare",
            "--strategies", "fifty-two-week-high",
            "--symbols", "SPY",
            "--snapshot", "t",
            "--out", str(out_csv),
            "--trades-dir", str(trades_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    # Caso 1: corrida normal coincide N de N y genera markdown si se pide
    out_md = tmp_path / "ctrl.md"
    _run_control(out_csv, trades_dir, "t", md=out_md)
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "coinciden 6 de 6"
    assert out_md.exists()
    ctrl_md_text = out_md.read_text(encoding="utf-8")
    assert "coinciden 6 de 6" in ctrl_md_text
    assert "- **Snapshot:** t" in ctrl_md_text

    # Caso 2: alterando un trade del motor (pnl en fila 1) da sin validar nombrando fila y columna
    motor_file = trades_dir / "fifty-two-week-high-SPY.csv"
    with open(motor_file, newline="") as f:
        trade_rows = list(csv.DictReader(f))
    trade_rows[0]["pnl"] = str(float(trade_rows[0]["pnl"]) + 100.0)
    with open(motor_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows)

    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: fila 1, columna pnl"

    # Caso 3: alterando exit_reason en fila 2
    trade_rows[0]["pnl"] = str(float(trade_rows[0]["pnl"]) - 100.0)  # restaurar fila 1
    trade_rows[1]["exit_reason"] = "OTRO_MOTIVO"
    with open(motor_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows)

    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: fila 2, columna exit_reason"

    # Caso 4: distinta cantidad de trades
    with open(motor_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trade_rows[:-1])

    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: distinta cantidad de trades (motor=5, port=6)"

    # Caso 5: falta archivo de trades
    motor_file.unlink()
    _run_control(out_csv, trades_dir, "t")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["control"] == "sin validar: falta archivo fifty-two-week-high-SPY.csv"


def test_compare_control_sensibilidad_apertura_siguiente_gap(snap_env):
    runner, tmp_path = snap_env
    df = _daily_frame().iloc[:330].copy()
    close_signal = df.loc[df.index[320], "close"]
    gap = 10.0
    open_next = close_signal + gap
    df.loc[df.index[321], "open"] = open_next
    df.loc[df.index[321], "high"] = max(df.loc[df.index[321], "high"], open_next)

    write_snapshot("t_gap", {"SPY": df})
    out_csv = tmp_path / "cmp_gap.csv"
    trades_dir = tmp_path / "trades_gap"

    res = runner.invoke(
        backtest_group,
        [
            "compare",
            "--strategies", "fifty-two-week-high",
            "--symbols", "SPY",
            "--snapshot", "t_gap",
            "--out", str(out_csv),
            "--trades-dir", str(trades_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    _run_control(out_csv, trades_dir, "t_gap")
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["control"] == "coinciden 1 de 1"
    assert rows[0]["trades"] == "1"
    assert rows[0]["trades_apertura_siguiente"] == "1"

    pnl_cierre = float(rows[0]["pnl"])
    pnl_apertura = float(rows[0]["pnl_apertura_siguiente"])
    assert pnl_apertura < pnl_cierre

    with open(trades_dir / "fifty-two-week-high-SPY.csv", newline="") as f:
        trade = next(csv.DictReader(f))
    qty = float(trade["qty"])
    slippage = float(settings.TRADING_SLIPPAGE_PCT)
    effective_gap = gap * (1.0 + slippage)
    expected_diff = round(effective_gap * qty, 2)
    assert round(pnl_cierre - pnl_apertura, 2) == pytest.approx(expected_diff, abs=0.01)
