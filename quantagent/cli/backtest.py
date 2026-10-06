"""CLI commands for running deterministic backtests (D13-D18 of PLAN-30-DIAS)."""

from __future__ import annotations

import logging
from typing import Optional

import click
from sqlalchemy import select

from quantagent.backtesting.backtest import Backtest
from quantagent.backtesting.export import TradeRow, equity_to_csv, trades_to_csv
from quantagent.backtesting.fixtures import fixture_metadata, load_fixture
from quantagent.models import ActivePosition, Trade
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
        load_fixture(session, fixture_name)

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

        rows = []
        if out_path:
            trade_ids = select(ActivePosition.trade_id).where(
                ActivePosition.backtest_run_id == bt.backtest_run_id,
                ActivePosition.trade_id.is_not(None),
            )
            trades = (
                session.query(Trade, ActivePosition.stop_loss)
                .join(ActivePosition, ActivePosition.trade_id == Trade.id)
                .filter(Trade.id.in_(trade_ids))
                .order_by(Trade.opened_at)
                .all()
            )
            rows = [
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

    click.echo(f"Trades: {metrics.total_trades}")
    click.echo(f"Win rate: {metrics.win_rate:.2%}")
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
