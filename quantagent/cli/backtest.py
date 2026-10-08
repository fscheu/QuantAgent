"""CLI commands for running deterministic backtests (D13-D18 of PLAN-30-DIAS)."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Optional

import click
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantagent.backtesting.backtest import Backtest
from quantagent.backtesting.export import TradeRow, equity_to_csv, trades_to_csv
from quantagent.backtesting.fixtures import fixture_metadata, load_fixture
from quantagent.models import ActivePosition, MarketData, Trade
from quantagent.strategy.registry import build_strategy

from . import utils

# CLI-friendly short names for the deterministic strategies. LLMAgentStrategy is
# excluded: it needs live model credentials, so it doesn't fit a fixture-based,
# credential-free `backtest run`.
STRATEGY_ALIASES = {
    "rsi": "RSIMeanReversionStrategy",
    "fifty-two-week-high": "FiftyTwoWeekHighStrategy",
    "triple-screen": "TripleScreenStrategy",
}


def _hide_data_warnings(record: logging.LogRecord) -> bool:
    return getattr(record, "event_type", None) != "backtest_data_warning"


def _extract_trade_rows(session: Session, backtest_run_id: int) -> list[TradeRow]:
    trade_ids = select(ActivePosition.trade_id).where(
        ActivePosition.backtest_run_id == backtest_run_id,
        ActivePosition.trade_id.is_not(None),
    )
    trades = (
        session.query(Trade, ActivePosition.stop_loss)
        .join(ActivePosition, ActivePosition.trade_id == Trade.id)
        .filter(Trade.id.in_(trade_ids))
        .order_by(Trade.opened_at)
        .all()
    )
    return [
        TradeRow(
            entry_time=trade.opened_at,
            exit_time=trade.closed_at,
            symbol=trade.symbol,
            side=trade.side.value,
            qty=float(trade.quantity),
            entry_price=float(trade.entry_price),
            exit_price=float(trade.exit_price) if trade.exit_price is not None else None,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            pnl=float(trade.pnl) if trade.pnl is not None else None,
            exit_reason=trade.exit_signal,
        )
        for trade, stop_loss in trades
    ]


@click.group(help="Run deterministic backtests against versioned fixtures.")
def backtest_group() -> None:
    """Root command group for backtest operations."""


@backtest_group.command("run", help="Run a backtest against a versioned fixture.")
@click.option(
    "--strategy",
    "strategy_name",
    required=True,
    type=click.Choice(sorted(STRATEGY_ALIASES)),
    help="Strategy to run.",
)
@click.option(
    "--fixture",
    "fixture_name",
    required=True,
    help="Fixture name under tests/fixtures/ (without the .csv extension).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True),
    help="Write the trade log CSV to this path.",
)
@click.option(
    "--equity-out",
    "equity_out_path",
    type=click.Path(dir_okay=False, writable=True),
    help="Write the equity curve CSV (one row per candle) to this path.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Also show the engine's per-candle 'Insufficient data' messages on stderr.",
)
def run_backtest(
    strategy_name: str,
    fixture_name: str,
    out_path: Optional[str],
    equity_out_path: Optional[str],
    verbose: bool,
) -> None:
    try:
        meta = fixture_metadata(fixture_name)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Fixture not found: {fixture_name}") from exc

    with utils.session_scope() as session:
        existing_count = (
            session.query(MarketData)
            .filter(
                MarketData.symbol == meta.symbol,
                MarketData.timeframe == meta.timeframe,
                MarketData.timestamp >= meta.start_date,
                MarketData.timestamp <= meta.end_date,
            )
            .count()
        )
        if existing_count == 0:
            load_fixture(session, fixture_name)
        elif existing_count != meta.row_count:
            raise click.ClickException(
                f"La base tiene otros datos para {meta.symbol} {meta.timeframe} en el rango "
                f"del fixture ({existing_count} filas, se esperaban {meta.row_count}): "
                "usá una base limpia."
            )

        bt = Backtest(
            start_date=meta.start_date,
            end_date=meta.end_date,
            assets=[meta.symbol],
            timeframe=meta.timeframe,
            # Fixtures are versioned as continuous candles (every hour, no market-hours
            # gaps), so filtering to real trading hours would misalign against them.
            # offline_data avoids live yfinance calls for the lookback window before the
            # fixture's first row -- the whole point of a fixture-based run is to be
            # deterministic and network-free.
            config={"market_hours_filter": False, "offline_data": True},
            db_session=session,
            strategy=build_strategy(STRATEGY_ALIASES[strategy_name]),
        )
        # The lookback warm-up logs one "Insufficient data" warning per candle; keep the
        # output to the metric lines unless --verbose asks for them.
        engine_logger = logging.getLogger("quantagent.backtesting.backtest")
        if not verbose:
            engine_logger.addFilter(_hide_data_warnings)
        try:
            metrics = bt.run(name=f"cli-{strategy_name}-{fixture_name}")
        finally:
            engine_logger.removeFilter(_hide_data_warnings)

        slippage_pct = bt.order_manager.broker.slippage_pct
        rows = _extract_trade_rows(session, bt.backtest_run_id) if out_path else []

    click.echo(f"Trades: {metrics.total_trades}")
    click.echo(f"Win rate: {metrics.win_rate:.2%}")
    if metrics.profit_factor is None or math.isinf(metrics.profit_factor):
        click.echo("Profit factor: n/a")
    else:
        click.echo(f"Profit factor: {metrics.profit_factor:.2f}")
    click.echo(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    click.echo(f"Total PnL: {metrics.total_pnl:.2f}")
    click.echo(f"Slippage: {slippage_pct * 100:.2f}% por lado")

    if out_path:
        with open(out_path, "w", newline="") as f:
            f.write(trades_to_csv(rows))
        click.echo(f"Trade log written to {out_path}")

    if equity_out_path:
        with open(equity_out_path, "w", newline="") as f:
            f.write(equity_to_csv(bt.equity_curve))
        click.echo(
            f"Equity curve written to {equity_out_path} "
            f"(max drawdown: {metrics.max_drawdown:.6f})"
        )


@contextmanager
def _isolated_sqlite_db():
    import os
    import tempfile

    from quantagent import database, settings

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "verify.db")
        db_url = f"sqlite:///{db_path}"
        old_env = os.environ.get("DATABASE_URL")
        old_settings_url = settings.DATABASE_URL
        os.environ["DATABASE_URL"] = db_url
        settings.DATABASE_URL = db_url
        if hasattr(database, "_engine"):
            database._engine = None
        if hasattr(database, "_SessionLocal"):
            database._SessionLocal = None
        try:
            database.init_db()
            yield
        finally:
            if old_env is not None:
                os.environ["DATABASE_URL"] = old_env
            else:
                os.environ.pop("DATABASE_URL", None)
            settings.DATABASE_URL = old_settings_url
            if hasattr(database, "_engine"):
                database._engine = None
            if hasattr(database, "_SessionLocal"):
                database._SessionLocal = None


def _run_verify_pass(strategy_name: str, fixture_name: str, verbose: bool):
    meta = fixture_metadata(fixture_name)
    with _isolated_sqlite_db(), utils.session_scope() as session:
        load_fixture(session, fixture_name)
        bt = Backtest(
            start_date=meta.start_date,
            end_date=meta.end_date,
            assets=[meta.symbol],
            timeframe=meta.timeframe,
            config={"market_hours_filter": False, "offline_data": True},
            db_session=session,
            strategy=build_strategy(STRATEGY_ALIASES[strategy_name]),
        )
        engine_logger = logging.getLogger("quantagent.backtesting.backtest")
        if not verbose:
            engine_logger.addFilter(_hide_data_warnings)
        try:
            metrics = bt.run(name=f"verify-{strategy_name}-{fixture_name}")
        finally:
            engine_logger.removeFilter(_hide_data_warnings)

        slippage = bt.order_manager.broker.slippage_pct
        rows = _extract_trade_rows(session, bt.backtest_run_id)
        return metrics, slippage, trades_to_csv(rows)


@backtest_group.command("verify", help="Verify backtest reproducibility across two runs.")
@click.option(
    "--strategy",
    "strategy_name",
    required=True,
    type=click.Choice(sorted(STRATEGY_ALIASES)),
    help="Strategy to run.",
)
@click.option(
    "--fixture",
    "fixture_name",
    required=True,
    help="Fixture name under tests/fixtures/ (without the .csv extension).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Also show engine warnings on stderr.",
)
def verify_backtest(strategy_name: str, fixture_name: str, verbose: bool) -> None:
    import difflib

    try:
        fixture_metadata(fixture_name)
    except FileNotFoundError as exc:
        raise click.ClickException(f"Fixture not found: {fixture_name}") from exc

    m1, s1, csv1 = _run_verify_pass(strategy_name, fixture_name, verbose)
    m2, s2, csv2 = _run_verify_pass(strategy_name, fixture_name, verbose)

    diff_lines = []
    if csv1 != csv2:
        diff_lines.extend(
            difflib.unified_diff(
                csv1.splitlines(keepends=True),
                csv2.splitlines(keepends=True),
                fromfile="run1_trades.csv",
                tofile="run2_trades.csv",
            )
        )

    for field in [
        "total_trades",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "total_pnl",
        "max_drawdown",
    ]:
        v1, v2 = getattr(m1, field), getattr(m2, field)
        if v1 != v2:
            diff_lines.append(f"Metric mismatch '{field}': run1={v1} != run2={v2}\n")

    if s1 != s2:
        diff_lines.append(f"Slippage mismatch: run1={s1} != run2={s2}\n")

    if diff_lines:
        click.echo("".join(diff_lines), nl=False)
        raise click.ClickException("Backtest reproducibility check failed")

    click.echo("OK reproducible")
