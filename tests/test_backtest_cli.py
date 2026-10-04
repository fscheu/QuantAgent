"""End-to-end test for `quantagent backtest run` via CliRunner (D17 of PLAN-30-DIAS).

Uses tests/fixtures/spy-smoke.csv (120 hourly candles, same deterministic sine-wave
generator as spy-90d.csv) instead of the full 90-day fixture -- small enough to run
in a fraction of a second while still producing real trades through the full engine
(no mocking), so the suite doesn't pay ~60s per run just to exercise this command.
"""

import csv
import os
import subprocess
import sys

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine

import quantagent.database as database
from quantagent import settings
from quantagent.backtesting.export import COLUMNS
from quantagent.cli.backtest import backtest_group
from quantagent.models import Base


@pytest.fixture
def cli_runner(monkeypatch, tmp_path):
    """Provide CliRunner with an isolated, fully-migrated SQLite database.

    quantagent.cli.utils._ensure_database_url() caches a process-global
    engine/session keyed off settings.DATABASE_URL whenever it sees the
    DATABASE_URL env var change (so successive CLI invocations in the same
    process pick up a new target). That caching outlives this test unless we
    reset it: otherwise a later test that touches quantagent.database.SessionLocal()
    directly (e.g. test_logging_infrastructure.py) inherits this sqlite engine
    instead of the real configured DB.
    """
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    original_database_url = settings.DATABASE_URL
    try:
        yield CliRunner()
    finally:
        settings.DATABASE_URL = original_database_url
        database._engine = None
        database._SessionLocal = None


def test_backtest_run_exits_zero_and_prints_metrics(cli_runner):
    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )

    assert result.exit_code == 0, result.output
    assert "Trades:" in result.output
    assert "Win rate:" in result.output
    assert "Profit factor:" in result.output
    assert "Sharpe ratio:" in result.output
    assert "Total PnL:" in result.output


def test_backtest_run_writes_csv_matching_reported_trade_count(cli_runner, tmp_path):
    out_path = tmp_path / "trades.csv"

    result = cli_runner.invoke(
        backtest_group,
        ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()

    trades_line = next(
        line for line in result.output.splitlines() if line.startswith("Trades:")
    )
    reported_trades = int(trades_line.split(":")[1].strip())
    assert reported_trades > 0  # spy-smoke is tuned to guarantee at least one trade

    with out_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)

    assert header == COLUMNS
    assert len(data_rows) == reported_trades


def test_backtest_run_unknown_fixture_fails_cleanly(cli_runner):
    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "does-not-exist"]
    )

    assert result.exit_code != 0
    assert "Fixture not found" in result.output


def _run_cli_process(tmp_path, *extra_args):
    """Run the CLI as a real process so stderr noise (logging, warnings) is counted too."""
    env_db = f"sqlite:///{tmp_path / 'proc.db'}"
    Base.metadata.create_all(create_engine(env_db))
    out_path = tmp_path / "trades.csv"
    return subprocess.run(
        [sys.executable, "-m", "quantagent.cli", "backtest", "run", "--strategy", "rsi",
         "--fixture", "spy-smoke", "--out", str(out_path), *extra_args],
        env={**os.environ, "DATABASE_URL": env_db, "OPENAI_API_KEY": "dummy-not-used"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_backtest_run_output_is_only_metrics_lines(tmp_path):
    """stdout+stderr is exactly the 5 metric lines plus the --out line: no log or SAWarning noise."""
    proc = _run_cli_process(tmp_path)

    assert proc.returncode == 0, proc.stdout
    prefixes = [line.split(":")[0] for line in proc.stdout.splitlines()]
    assert prefixes == [
        "Trades", "Win rate", "Profit factor", "Sharpe ratio", "Total PnL",
        f"Trade log written to {tmp_path / 'trades.csv'}",
    ]


def test_backtest_run_verbose_shows_insufficient_data_messages(tmp_path):
    """--verbose brings back the engine's per-candle 'Insufficient data' messages."""
    proc = _run_cli_process(tmp_path, "--verbose")

    assert proc.returncode == 0, proc.stdout
    assert "Insufficient data for SPY" in proc.stdout
    assert "Trades: " in proc.stdout
