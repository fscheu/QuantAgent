"""End-to-end test for `quantagent backtest run` via CliRunner (D17 of PLAN-30-DIAS).

Uses tests/fixtures/spy-smoke.csv (120 hourly candles, same deterministic sine-wave
generator as spy-90d.csv) instead of the full 90-day fixture -- small enough to run
in a fraction of a second while still producing real trades through the full engine
(no mocking), so the suite doesn't pay ~60s per run just to exercise this command.
"""

import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import quantagent.database as database
from quantagent import settings
from quantagent.backtesting.export import COLUMNS, EQUITY_COLUMNS
from quantagent.backtesting.fixtures import FIXTURES_DIR, fixture_metadata
from quantagent.cli.backtest import backtest_group
from quantagent.models import Base, MarketData


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


@pytest.fixture(scope="module")
def shared_rsi_run(tmp_path_factory):
    """Run `backtest run` with RSI on spy-smoke once for all tests verifying its outputs."""
    tmp_path = tmp_path_factory.mktemp("shared_rsi")
    db_path, out_path = tmp_path / "test.db", tmp_path / "trades.csv"
    Base.metadata.create_all(create_engine(f"sqlite:///{db_path}"))
    old_env, old_url = os.environ.get("DATABASE_URL"), settings.DATABASE_URL
    os.environ["DATABASE_URL"] = settings.DATABASE_URL = f"sqlite:///{db_path}"
    database._engine = database._SessionLocal = None
    try:
        res = CliRunner().invoke(
            backtest_group,
            ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--out", str(out_path)],
        )
    finally:
        if old_env is not None:
            os.environ["DATABASE_URL"] = old_env
        else:
            os.environ.pop("DATABASE_URL", None)
        settings.DATABASE_URL = old_url
        database._engine = database._SessionLocal = None
    return res, out_path, db_path


def test_backtest_run_exits_zero_and_prints_metrics(cli_runner):
    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )

    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 14
    assert lines[0].startswith("Trades:")
    assert lines[1].startswith("Win rate:")
    assert lines[2].startswith("Profit factor:")
    assert lines[3].startswith("Sharpe ratio:")
    assert lines[4].startswith("Total PnL:")
    assert lines[5] == "Slippage: 0.05% por lado"
    assert lines[6] == "Comisión: 0.00% por lado"
    assert lines[7].startswith("Comprar y mantener: PnL ")
    assert [line.split(":")[0] for line in lines[8:]] == [
        "Retorno anualizado", "Exceso sobre la referencia", "Tiempo en mercado", "Capital invertido medio",
        "Sharpe (tasa 0)", "Peor día",
    ]
    assert lines[13] == "Peor día: 0.00%"


def test_backtest_run_commission_pct_charges_every_fill_and_lowers_pnl(cli_runner):
    """--commission-pct llega al broker con modelo pct: la entrada queda registrada como qty * precio * pct
    en el trade y la salida baja el Total PnL, sin cambiar los trades (QuantAgent-de5)."""
    args = ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    base = cli_runner.invoke(backtest_group, args)
    assert base.exit_code == 0, base.output
    result = cli_runner.invoke(backtest_group, [*args, "--commission-pct", "0.001"])
    assert result.exit_code == 0, result.output

    out0 = dict(line.split(": ", 1) for line in base.output.splitlines())
    out1 = dict(line.split(": ", 1) for line in result.output.splitlines())
    assert out1["Comisión"] == "0.10% por lado"
    assert out1["Trades"] == out0["Trades"]
    assert float(out1["Total PnL"]) < float(out0["Total PnL"])

    with Session(create_engine(os.environ["DATABASE_URL"])) as s:
        rows = s.execute(text(
            "SELECT quantity, entry_price, commission FROM trades"
            " WHERE backtest_run_id = (SELECT MAX(id) FROM backtest_runs)"
        )).all()
    assert len(rows) == int(out1["Trades"])
    for qty, entry, commission in rows:
        assert float(commission) == pytest.approx(float(qty) * float(entry) * 0.001, rel=1e-9)


def test_backtest_run_writes_csv_matching_reported_trade_count(shared_rsi_run):
    result, out_path, _ = shared_rsi_run

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


def test_backtest_run_fifty_two_week_high_trades_on_daily_fixture(cli_runner, tmp_path):
    out_path = tmp_path / "trades.csv"

    result = cli_runner.invoke(
        backtest_group,
        ["run", "--strategy", "fifty-two-week-high", "--fixture", "spy-2y-1d", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    reported_trades = int(result.output.splitlines()[0].split(":")[1])
    assert reported_trades >= 5

    with out_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (FIXTURES_DIR / "spy-2y-1d.csv").open(newline="") as f:
        fixture_close = {r["timestamp"]: float(r["close"]) for r in csv.DictReader(f)}

    assert len(rows) == reported_trades
    for row in rows:
        assert row["entry_time"] in fixture_close
        assert float(row["entry_price"]) == pytest.approx(fixture_close[row["entry_time"]], rel=0.001)


def test_backtest_run_triple_screen_trades_on_4h_fixture(cli_runner, tmp_path):
    out_path = tmp_path / "trades.csv"

    result = cli_runner.invoke(
        backtest_group,
        ["run", "--strategy", "triple-screen", "--fixture", "spy-1y-4h", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    reported_trades = int(result.output.splitlines()[0].split(":")[1])
    assert reported_trades >= 5

    with out_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (FIXTURES_DIR / "spy-1y-4h.csv").open(newline="") as f:
        fixture_close = {r["timestamp"]: float(r["close"]) for r in csv.DictReader(f)}

    assert len(rows) == reported_trades
    for row in rows:
        assert row["entry_time"] in fixture_close
        assert float(row["entry_price"]) == pytest.approx(fixture_close[row["entry_time"]], rel=0.001)


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
    # The equity CSV is rounded to 4 decimals, so the match is only that precise.
    assert max(float(r["drawdown_pct"]) for r in rows) == pytest.approx(engine_dd, abs=1e-4)
    assert f"(max drawdown: {engine_dd:.6f})" in result.output
    equities = [float(r["equity"]) for r in rows]
    assert equities[-1] - 100000.0 == pytest.approx(captured["metrics"].total_pnl, abs=0.01)
    assert all(abs(b - a) / a <= 0.02 for a, b in zip(equities, equities[1:]))


def test_backtest_run_stores_one_closed_trade_row_per_round_trip(shared_rsi_run):
    """Closed Trade rows of a run must equal its round-trips, and their pnl must add up to Total PnL.

    Regression for QuantAgent-89e: every close (take-profit, stop-loss, end of backtest) used to
    leave two closed rows with the same pnl, the opening trade and its closing leg.
    """
    result, _, db_path = shared_rsi_run
    assert result.exit_code == 0, result.output
    reported = dict(line.split(": ") for line in result.output.splitlines() if ": " in line)

    with create_engine(f"sqlite:///{db_path}").connect() as conn:
        round_trips = conn.execute(
            text("SELECT COUNT(*) FROM active_positions WHERE is_active = 0")
        ).scalar()
        closed_rows, pnl_sum = conn.execute(
            text("SELECT COUNT(*), SUM(pnl) FROM trades WHERE closed_at IS NOT NULL")
        ).one()

    assert round_trips == int(reported["Trades"]) > 0
    assert closed_rows == round_trips
    assert float(pnl_sum) == pytest.approx(float(reported["Total PnL"]), abs=0.01)


def test_backtest_run_trade_exit_price_is_executed_not_theoretical(shared_rsi_run):
    """The CSV exit_price is the executed fill price with slippage, not the theoretical candle price.

    Regression for QuantAgent-hx0.8: previously exit_price had no slippage while entry_price did,
    so (exit - entry) * qty diverged from the realized pnl on both long and short trades.
    """
    result, out_path, _ = shared_rsi_run
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


@pytest.fixture(scope="module")
def shared_cli_process_run(tmp_path_factory):
    """Run the CLI subprocess once to verify clean output format and OpenAI key independence."""
    tmp_path = tmp_path_factory.mktemp("cli_proc")
    proc = _run_cli_process(tmp_path)
    return proc, tmp_path / "trades.csv"


def test_backtest_run_output_is_only_metrics_lines(shared_cli_process_run):
    """stdout+stderr is exactly the 14 metric lines plus the --out line: no log or SAWarning noise."""
    proc, trades_csv = shared_cli_process_run

    assert proc.returncode == 0, proc.stdout
    prefixes = [line.split(":")[0] for line in proc.stdout.splitlines()]
    assert prefixes == [
        "Trades", "Win rate", "Profit factor", "Sharpe ratio", "Total PnL", "Slippage", "Comisión",
        "Comprar y mantener", "Retorno anualizado", "Exceso sobre la referencia", "Tiempo en mercado",
        "Capital invertido medio", "Sharpe (tasa 0)", "Peor día",
        f"Trade log written to {trades_csv}",
    ]
    assert proc.stdout.splitlines()[5] == "Slippage: 0.05% por lado"
    assert proc.stdout.splitlines()[6] == "Comisión: 0.00% por lado"


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
    assert len(lines) == 14
    assert lines[-9] == "Slippage: 1.00% por lado"


def test_backtest_run_verbose_shows_insufficient_data_messages(tmp_path):
    """--verbose brings back the engine's per-candle 'Insufficient data' messages."""
    proc = _run_cli_process(tmp_path, "--verbose")

    assert proc.returncode == 0, proc.stdout
    assert "Insufficient data for SPY" in proc.stdout
    assert "Trades: " in proc.stdout


def test_backtest_run_deterministic_strategy_needs_no_openai_key(shared_cli_process_run):
    """Without OPENAI_API_KEY the rsi backtest still runs: it must not build the LLM client."""
    proc, _ = shared_cli_process_run

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


@pytest.fixture(scope="module")
def shared_verify_smoke():
    """Run `backtest verify` once per strategy on spy-smoke and share across verify tests."""
    runner = CliRunner()
    return {
        strat: runner.invoke(
            backtest_group, ["verify", "--strategy", strat, "--fixture", "spy-smoke"]
        )
        for strat in ["rsi", "fifty-two-week-high", "triple-screen"]
    }


def test_backtest_verify_success_exits_zero_and_prints_ok_reproducible(shared_verify_smoke):
    """Verify RSI strategy against spy-smoke fixture passes and prints OK reproducible."""
    result = shared_verify_smoke["rsi"]
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "OK reproducible"


def test_backtest_verify_all_three_strategies(shared_verify_smoke):
    """Verify accepts all 3 deterministic strategies (RSI, Fifty-Two-Week-High, Triple Screen)."""
    for strat in ["rsi", "fifty-two-week-high", "triple-screen"]:
        result = shared_verify_smoke[strat]
        assert result.exit_code == 0, f"Strategy {strat} failed: {result.output}"
        assert "OK reproducible" in result.output


def test_backtest_verify_detects_difference_and_exits_nonzero(cli_runner, monkeypatch):
    """When two runs produce different metrics, verify outputs the mismatch and exits non-zero."""
    import copy

    from quantagent.cli import backtest as bt_cli
    real_execute = bt_cli._run_verify_pass
    first_pass = None

    def patched_execute(*args, **kwargs):
        nonlocal first_pass
        if first_pass is None:
            first_pass = real_execute(*args, **kwargs)
            return first_pass
        metrics, slippage, csv_str = first_pass
        m2 = copy.copy(metrics)
        m2.total_trades += 1
        return m2, slippage, csv_str

    monkeypatch.setattr(bt_cli, "_run_verify_pass", patched_execute)

    result = cli_runner.invoke(
        backtest_group, ["verify", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )
    assert result.exit_code != 0
    assert "Metric mismatch 'total_trades'" in result.output
    assert "OK reproducible" not in result.output


def test_backtest_verify_unknown_fixture_fails_cleanly(cli_runner):
    """Verify reports fixture not found cleanly."""
    result = cli_runner.invoke(
        backtest_group, ["verify", "--strategy", "rsi", "--fixture", "nonexistent"]
    )
    assert result.exit_code != 0
    assert "Fixture not found" in result.output


def test_backtest_verify_process_output_is_only_ok_reproducible():
    """Real process invocation prints only 'OK reproducible' with zero exit code."""
    proc = subprocess.run(
        [sys.executable, "-m", "quantagent.cli", "backtest", "verify", "--strategy", "rsi",
         "--fixture", "spy-smoke"],
        env={**os.environ, **NO_OPENAI_KEY_ENV},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip() == "OK reproducible"


def test_backtest_run_consecutive_runs_on_same_database_produce_identical_output(
    cli_runner,
):
    """Two consecutive `backtest run` on the same database must produce identical 14 lines (QuantAgent-hx0.11)."""
    res1 = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )
    assert res1.exit_code == 0, res1.output
    lines1 = res1.output.strip().splitlines()
    assert len(lines1) == 14

    res2 = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )
    assert res2.exit_code == 0, res2.output
    lines2 = res2.output.strip().splitlines()
    assert len(lines2) == 14

    assert lines1 == lines2


def test_backtest_run_aborts_without_deleting_foreign_market_data(cli_runner):
    """Si la base ya tiene otras velas en el rango, aborta y no las borra (QuantAgent-hx0.11)."""
    meta = fixture_metadata("spy-smoke")
    engine = create_engine(os.environ["DATABASE_URL"])
    with Session(engine) as s:
        s.add_all(
            MarketData(
                symbol=meta.symbol,
                timeframe=meta.timeframe,
                timestamp=meta.start_date + timedelta(hours=i),
                open=1,
                high=1,
                low=1,
                close=1,
                volume=1,
            )
            for i in range(5)
        )
        s.commit()

    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )

    assert result.exit_code != 0
    assert "base limpia" in result.output
    with Session(engine) as s:
        assert s.query(MarketData).count() == 5


@pytest.mark.parametrize("pf_val", [float("inf"), None])
def test_backtest_run_prints_na_profit_factor_when_no_losses(cli_runner, monkeypatch, pf_val):
    """When a run has no losing trades (profit factor inf or None), CLI prints 'Profit factor: n/a'."""
    from quantagent.backtesting.backtest import Backtest, BacktestMetrics

    def mock_run(self, *args, **kwargs):
        return BacktestMetrics(
            total_trades=2,
            winning_trades=2,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=pf_val,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            total_pnl=500.0,
            avg_win=250.0,
            avg_loss=0.0,
            largest_win=300.0,
            largest_loss=0.0,
            total_return_pct=5.0,
        )

    monkeypatch.setattr(Backtest, "run", mock_run)
    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[2] == "Profit factor: n/a"


def test_annual_return_and_exposure_on_a_hand_curve_and_edge_cases():
    """Validates the engine-side formulas (QuantAgent-ol11) against values computed here, plus the degenerate curves."""
    from quantagent.backtesting.buy_and_hold import annual_return, exposure

    equity, pv = [100000.0, 101000.0, 100500.0, 102000.0, 103000.0], [0.0, 40400.0, -50250.0, 0.0, 25750.0]
    curve = [{"equity": e, "positions_value": v} for e, v in zip(equity, pv)]
    t0, t1 = datetime(2025, 1, 1), datetime(2025, 1, 5)
    assert annual_return(equity[0], equity[-1], t0, t1) == pytest.approx(1.03 ** (365.25 / 4) - 1)
    in_market, invested = exposure(curve)
    assert in_market == pytest.approx(3 / 5)
    assert invested == pytest.approx((0.4 + 0.5 + 0.25) / 5)
    assert annual_return(100.0, 105.0, t0, t0) == 0.0  # ventana de 0 días
    assert exposure(curve[:1]) == (0.0, 0.0)  # una sola vela, sin posición


def test_backtest_run_empty_equity_curve_omits_comparison_metrics(cli_runner, monkeypatch):
    """When the equity curve is empty, the 5 comparison metrics are omitted and command exits 0 (QuantAgent-ol11)."""
    from quantagent.backtesting.backtest import Backtest

    monkeypatch.setattr(Backtest, "_record_equity", lambda self, *a, **k: None)
    result = cli_runner.invoke(
        backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"]
    )
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 8
    assert not any(
        line.startswith(
            ("Retorno anualizado", "Exceso sobre la referencia", "Tiempo en mercado", "Capital invertido medio", "Sharpe (tasa 0)", "Peor día")
        )
        for line in lines
    )


@pytest.mark.parametrize("extra, pct, regla", [
    (["--tamano-posicion", "1.0"], 1.0, "confianza"),
    (["--regla-tamano", "fijo", "--tamano-posicion", "0.5"], 0.5, "fijo"),
])
def test_backtest_run_tamano_posicion_y_regla_dimensionan_cada_entrada(cli_runner, tmp_path, monkeypatch, extra, pct, regla):
    """--tamano-posicion y --regla-tamano llegan al motor (QuantAgent-3has): en los 3 primeros trades
    qty = equity al entrar × fracción (× confianza de la señal, salvo regla fijo) / precio de entrada."""
    monkeypatch.setattr(settings, "TRADING_SLIPPAGE_PCT", 0.0)  # sin slippage, precio de entrada = precio de la señal
    out = tmp_path / "trades.csv"
    result = cli_runner.invoke(backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--out", str(out), *extra])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 16
    assert f"Tamaño: {pct:.0%} del capital, regla {regla}" in lines
    assert [line.split(":")[0] for line in lines[9:15]] == [
        "Retorno anualizado", "Exceso sobre la referencia", "Tiempo en mercado", "Capital invertido medio",
        "Sharpe (tasa 0)", "Peor día",
    ]
    with Session(create_engine(os.environ["DATABASE_URL"])) as s:
        senales = s.execute(text("SELECT generated_at, confidence FROM signals ORDER BY generated_at LIMIT 3")).all()
        config = s.execute(text("SELECT config_snapshot FROM backtest_runs")).scalar_one()
    assert all(f'"{k}": {v}' in config for k, v in [("base_position_pct", pct), ("max_position_pct", pct), ("regla_tamano", f'"{regla}"')])
    trades = list(csv.DictReader(out.read_text().splitlines()))[:3]
    assert len(trades) == 3 and len({c for _, c in senales}) == 3  # confianzas distintas: las dos reglas no se confunden
    equity = 100000.0
    for trade, (generada, confianza) in zip(trades, senales):
        assert str(generada)[:19].replace(" ", "T") == trade["entry_time"]
        factor = confianza if regla == "confianza" else 1.0
        assert float(trade["qty"]) == pytest.approx(equity * pct * factor / float(trade["entry_price"]), rel=1e-7)
        equity += float(trade["pnl"])  # una posición por vez: el equity al entrar es capital + PnL ya realizado


def test_backtest_verify_usa_el_mismo_tamano_y_regla_en_las_dos_corridas(cli_runner, monkeypatch):
    """verify pasa --tamano-posicion y --regla-tamano a las dos corridas: en ambas el primer trade usa equity × 0.5 / precio."""
    from quantagent.cli import backtest as bt_cli
    real, csvs = bt_cli._run_verify_pass, []
    monkeypatch.setattr(bt_cli, "_run_verify_pass", lambda *a, **k: csvs.append(real(*a, **k)) or csvs[-1])
    monkeypatch.setattr(settings, "TRADING_SLIPPAGE_PCT", 0.0)
    result = cli_runner.invoke(
        backtest_group, ["verify", "--strategy", "rsi", "--fixture", "spy-smoke", "--regla-tamano", "fijo", "--tamano-posicion", "0.5"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["Tamaño: 50% del capital, regla fijo", "OK reproducible"]
    assert len(csvs) == 2
    for _, _, trades_csv in csvs:
        primero = next(csv.DictReader(trades_csv.splitlines()))
        assert float(primero["qty"]) == pytest.approx(100000.0 * 0.5 / float(primero["entry_price"]), rel=1e-7)


@pytest.mark.parametrize("cmd", ["run", "verify"])
@pytest.mark.parametrize("extra", [["--tamano-posicion", "0"], ["--tamano-posicion", "1.01"], ["--regla-tamano", "kelly"]])
def test_backtest_tamano_o_regla_invalidos_salen_con_2_sin_traceback(cli_runner, cmd, extra):
    result = cli_runner.invoke(backtest_group, [cmd, "--strategy", "rsi", "--fixture", "spy-smoke", *extra])
    assert result.exit_code == 2
    assert f"Invalid value for '{extra[0]}'" in result.output and "Traceback" not in result.output


def test_correr_par_devuelve_los_numeros_del_resumen_con_las_claves_documentadas(cli_runner):
    """QuantAgent-5nx0: rsi/spy-90d da 340 trades y PnL 4571.27 (las cifras de `backtest run`), el dict
    trae todas las claves del docstring y las cinco métricas de comparación vienen calculadas."""
    from quantagent.cli.backtest import _engine_config, correr_par

    r = correr_par("rsi", fixture_metadata("spy-90d"), _engine_config(None), None, "spy-90d", None, None, None, None)

    assert r["trades"] == 340
    assert round(r["pnl"], 2) == 4571.27
    assert all(re.search(rf"\b{k}\b", correr_par.__doc__) for k in r)
    assert (r["slippage_pct"], r["comision_pct"], r["costos"], r["tamano_pct"]) == (0.0005, 0.0, None, None)
    assert all(r[k] is not None for k in (
        "retorno_anualizado", "exceso_sobre_referencia", "tiempo_en_mercado", "capital_invertido_medio", "sharpe_tasa_0", "peor_dia",
    ))
    assert r["filas"] == []


def test_worst_day_on_a_hand_curve_and_edge_cases():
    """Validates engine-side worst_day on 5 days (one with 2 candles) and degenerate cases (QuantAgent-6zam)."""
    from datetime import datetime

    from quantagent.backtesting.buy_and_hold import worst_day

    # 5 days, day 3 has 2 candles: 100k -> 101k -> (99k, 98k) -> 99.5k -> 99k
    curve = [
        {"date": datetime(2025, 1, 1, 0, 0), "equity": 100000.0},
        {"date": datetime(2025, 1, 2, 0, 0), "equity": 101000.0},
        {"date": datetime(2025, 1, 3, 10, 0), "equity": 99000.0},
        {"date": datetime(2025, 1, 3, 16, 0), "equity": 98000.0},
        {"date": datetime(2025, 1, 4, 0, 0), "equity": 99500.0},
        {"date": datetime(2025, 1, 5, 0, 0), "equity": 99000.0},
    ]
    pct, fecha = worst_day(curve)
    assert pct == pytest.approx(-3000 / 101000)
    assert fecha == "2025-01-03"

    # All days rising -> 0.0, None
    up = [{"date": datetime(2025, 1, i), "equity": 100.0 + i} for i in range(1, 6)]
    assert worst_day(up) == (0.0, None)

    # Empty and single point
    assert worst_day([]) == (0.0, None)
    assert worst_day([{"date": datetime(2025, 1, 1), "equity": 100.0}]) == (0.0, None)


def test_isolated_sqlite_db_synchronous_off_and_external_default():
    """Dentro de _isolated_sqlite_db() PRAGMA synchronous = 0; fuera, una SQLite común conserva su valor (QuantAgent-06yj)."""
    from quantagent.cli.backtest import _isolated_sqlite_db
    import quantagent.database as db

    with _isolated_sqlite_db():
        with db.engine.connect() as conn:
            assert conn.execute(text("PRAGMA synchronous")).scalar() == 0

    other_engine = create_engine("sqlite:///:memory:")
    with other_engine.connect() as conn:
        assert conn.execute(text("PRAGMA synchronous")).scalar() == 2


def test_isolated_sqlite_db_fallback_without_shm(monkeypatch):
    """Con /dev/shm no disponible (simulado), la base se crea igual y la corrida termina (QuantAgent-06yj)."""
    import os
    from quantagent.cli.backtest import _isolated_sqlite_db
    import quantagent.database as db

    original_isdir = os.path.isdir
    monkeypatch.setattr(os.path, "isdir", lambda path: False if path == "/dev/shm" else original_isdir(path))

    with _isolated_sqlite_db():
        with db.engine.connect() as conn:
            assert conn.execute(text("PRAGMA synchronous")).scalar() == 0
            assert "/dev/shm" not in db.settings.DATABASE_URL


def test_isolated_sqlite_db_no_leftover_files_in_shm():
    """No quedan archivos en /dev/shm al terminar, tampoco si una corrida falla (QuantAgent-06yj)."""
    import os
    from quantagent.cli.backtest import _isolated_sqlite_db

    if not (os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)):
        pytest.skip("/dev/shm no está disponible")

    before = set(os.listdir("/dev/shm"))
    with _isolated_sqlite_db():
        pass
    assert set(os.listdir("/dev/shm")) == before

    with pytest.raises(RuntimeError):
        with _isolated_sqlite_db():
            raise RuntimeError("error simulado")
    assert set(os.listdir("/dev/shm")) == before


def test_backtest_run_orden_minima_sin_opcion_no_imprime_linea(cli_runner):
    """Sin --orden-minima, no se imprime la línea de descartadas y el resultado es idéntico (QuantAgent-wquk)."""
    base = cli_runner.invoke(backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke"])
    assert base.exit_code == 0, base.output
    assert "Entradas descartadas por orden mínima:" not in base.output

    con_min = cli_runner.invoke(backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--orden-minima", "50"])
    assert con_min.exit_code == 0, con_min.output
    assert "Entradas descartadas por orden mínima: 0" in con_min.output

    out_base = dict(line.split(": ", 1) for line in base.output.splitlines() if ": " in line)
    out_min = dict(line.split(": ", 1) for line in con_min.output.splitlines() if ": " in line)
    assert out_base["Trades"] == out_min["Trades"]
    assert out_base["Total PnL"] == out_min["Total PnL"]


def test_backtest_run_orden_minima_descarta_entradas(cli_runner):
    """Con --orden-minima superior al tamaño de las órdenes, las entradas se descartan (QuantAgent-wquk)."""
    # En spy-smoke el capital es 100k y base 5% -> órdenes de ~2500 a 5000 USD
    # Con --orden-minima 10000 se descartan todas las entradas
    res = cli_runner.invoke(backtest_group, ["run", "--strategy", "rsi", "--fixture", "spy-smoke", "--orden-minima", "10000"])
    assert res.exit_code == 0, res.output
    out = dict(line.split(": ", 1) for line in res.output.splitlines() if ": " in line)
    assert int(out["Trades"]) == 0
    assert int(out["Entradas descartadas por orden mínima"]) > 0
