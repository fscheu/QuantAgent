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
from sqlalchemy import create_engine, text

import quantagent.database as database
from quantagent import settings
from quantagent.backtesting.export import COLUMNS, EQUITY_COLUMNS
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
    lines = result.output.strip().splitlines()
    assert len(lines) == 6
    assert lines[0].startswith("Trades:")
    assert lines[1].startswith("Win rate:")
    assert lines[2].startswith("Profit factor:")
    assert lines[3].startswith("Sharpe ratio:")
    assert lines[4].startswith("Total PnL:")
    assert lines[5] == "Slippage: 0.05% por lado"


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


def test_backtest_run_equity_out_max_drawdown_matches_engine(cli_runner, tmp_path, monkeypatch):
    """--equity-out writes one row per equity point; its max drawdown_pct is the engine's max_drawdown."""
    from quantagent.backtesting.backtest import Backtest

    captured = {}
    original_run = Backtest.run

    def spy_run(self, *args, **kwargs):
        captured["metrics"] = original_run(self, *args, **kwargs)
        captured["points"] = len(self.equity_curve)
        return captured["metrics"]

    monkeypatch.setattr(Backtest, "run", spy_run)
    eq_path = tmp_path / "eq.csv"

    result = cli_runner.invoke(
        backtest_group,
        ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--equity-out", str(eq_path)],
    )

    assert result.exit_code == 0, result.output
    with eq_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == EQUITY_COLUMNS
    assert len(rows) == captured["points"] > 1
    engine_dd = captured["metrics"].max_drawdown
    assert engine_dd > 0  # spy-smoke trades, so the curve must dip at least once
    assert max(float(r["drawdown_pct"]) for r in rows) == pytest.approx(engine_dd, abs=1e-9)
    assert f"(max drawdown: {engine_dd:.6f})" in result.output
    equities = [float(r["equity"]) for r in rows]
    assert equities[-1] - 100000.0 == pytest.approx(captured["metrics"].total_pnl, abs=0.01)
    assert all(abs(b - a) / a <= 0.02 for a, b in zip(equities, equities[1:]))


def test_backtest_run_stores_one_closed_trade_row_per_round_trip(cli_runner, tmp_path):
    """Closed Trade rows of a run must equal its round-trips, and their pnl must add up to Total PnL.

    Regression for QuantAgent-89e: every close (take-profit, stop-loss, end of backtest) used to
    leave two closed rows with the same pnl, the opening trade and its closing leg.
    """
    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )
    assert result.exit_code == 0, result.output
    reported = dict(line.split(": ") for line in result.output.splitlines() if ": " in line)

    with create_engine(f"sqlite:///{tmp_path / 'test.db'}").connect() as conn:
        round_trips = conn.execute(
            text("SELECT COUNT(*) FROM active_positions WHERE is_active = 0")
        ).scalar()
        closed_rows, pnl_sum = conn.execute(
            text("SELECT COUNT(*), SUM(pnl) FROM trades WHERE closed_at IS NOT NULL")
        ).one()

    assert round_trips == int(reported["Trades"]) > 0
    assert closed_rows == round_trips
    assert float(pnl_sum) == pytest.approx(float(reported["Total PnL"]), abs=0.01)


def test_backtest_run_trade_exit_price_is_executed_not_theoretical(cli_runner, tmp_path):
    """The CSV exit_price is the executed fill price with slippage, not the theoretical candle price.

    Regression for QuantAgent-hx0.8: previously exit_price had no slippage while entry_price did,
    so (exit - entry) * qty diverged from the realized pnl on both long and short trades.
    """
    out_path = tmp_path / "trades.csv"
    result = cli_runner.invoke(
        backtest_group,
        ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output

    with out_path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["pnl"]]

    long_rows = [r for r in rows if r["side"] == "buy"]
    short_rows = [r for r in rows if r["side"] == "sell"]
    assert len(long_rows) > 0 and len(short_rows) > 0

    for r in rows:
        entry, exit_, qty, pnl = float(r["entry_price"]), float(r["exit_price"]), float(r["qty"]), float(r["pnl"])
        calc_pnl = (exit_ - entry) * qty if r["side"] == "buy" else (entry - exit_) * qty
        assert abs(calc_pnl - pnl) <= 0.01

        # With default slippage, executed exit diverges from theoretical candle price.
        # If exit_price reverted to theoretical, (theoretical - entry) * qty would diverge from pnl.
        slippage = settings.TRADING_SLIPPAGE_PCT
        theoretical = exit_ / (1 - slippage) if r["side"] == "buy" else exit_ / (1 + slippage)
        theoretical_calc = (theoretical - entry) * qty if r["side"] == "buy" else (entry - theoretical) * qty
        assert abs(theoretical_calc - pnl) > 0.1


# Blank (not unset) so a developer's .env cannot put the key back via load_dotenv.
NO_OPENAI_KEY_ENV = {"OPENAI_API_KEY": "", "AGENT_LLM_PROVIDER": "openai", "GRAPH_LLM_PROVIDER": "openai"}


def _run_cli_process(tmp_path, *extra_args):
    """Run the CLI as a real process so stderr noise (logging, warnings) is counted too."""
    env_db = f"sqlite:///{tmp_path / 'proc.db'}"
    Base.metadata.create_all(create_engine(env_db))
    out_path = tmp_path / "trades.csv"
    return subprocess.run(
        [sys.executable, "-m", "quantagent.cli", "backtest", "run", "--strategy", "rsi",
         "--fixture", "spy-smoke", "--out", str(out_path), *extra_args],
        env={**os.environ, "DATABASE_URL": env_db, **NO_OPENAI_KEY_ENV},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def test_backtest_run_output_is_only_metrics_lines(tmp_path):
    """stdout+stderr is exactly the 6 metric lines plus the --out line: no log or SAWarning noise."""
    proc = _run_cli_process(tmp_path)

    assert proc.returncode == 0, proc.stdout
    prefixes = [line.split(":")[0] for line in proc.stdout.splitlines()]
    assert prefixes == [
        "Trades", "Win rate", "Profit factor", "Sharpe ratio", "Total PnL", "Slippage",
        f"Trade log written to {tmp_path / 'trades.csv'}",
    ]
    assert proc.stdout.splitlines()[5] == "Slippage: 0.05% por lado"


def test_backtest_run_output_with_custom_slippage_env(tmp_path):
    """Setting TRADING_SLIPPAGE_PCT changes the effective slippage line."""
    env_db = f"sqlite:///{tmp_path / 'proc.db'}"
    Base.metadata.create_all(create_engine(env_db))
    proc = subprocess.run(
        [sys.executable, "-m", "quantagent.cli", "backtest", "run", "--strategy", "rsi",
         "--fixture", "spy-smoke"],
        env={**os.environ, "DATABASE_URL": env_db, "TRADING_SLIPPAGE_PCT": "0.01", **NO_OPENAI_KEY_ENV},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 6
    assert lines[-1] == "Slippage: 1.00% por lado"


def test_backtest_run_verbose_shows_insufficient_data_messages(tmp_path):
    """--verbose brings back the engine's per-candle 'Insufficient data' messages."""
    proc = _run_cli_process(tmp_path, "--verbose")

    assert proc.returncode == 0, proc.stdout
    assert "Insufficient data for SPY" in proc.stdout
    assert "Trades: " in proc.stdout


def test_backtest_run_deterministic_strategy_needs_no_openai_key(tmp_path):
    """Without OPENAI_API_KEY the rsi backtest still runs: it must not build the LLM client."""
    proc = _run_cli_process(tmp_path)

    assert proc.returncode == 0, proc.stdout
    assert "OPENAI_API_KEY" not in proc.stdout
    assert proc.stdout.startswith("Trades: ")


def test_backtest_default_llm_strategy_still_requires_openai_key(tmp_path):
    """Without a strategy the engine defaults to LLMAgentStrategy, which must still demand the key."""
    env_db = f"sqlite:///{tmp_path / 'llm.db'}"
    Base.metadata.create_all(create_engine(env_db))
    code = (
        "from datetime import datetime\n"
        "from quantagent.backtesting.backtest import Backtest\n"
        "Backtest(datetime(2024, 1, 1), datetime(2024, 1, 2), ['SPY'], timeframe='1h')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "DATABASE_URL": env_db, **NO_OPENAI_KEY_ENV},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert proc.returncode != 0
    assert "OPENAI_API_KEY not found" in proc.stdout
