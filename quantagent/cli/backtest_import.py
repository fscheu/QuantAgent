"""Import saved research curves without rerunning the backtest engine."""

import math
import os
from pathlib import Path

import click
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from quantagent.fabrica.investigador import HIPOTESIS_DIR
from quantagent.models import BacktestRun

from .backtest_vincular import known_hypothesis_ids, resolve_hypothesis_id


@click.command(
    "import", help="Import every equity/*.csv below ROOT (default: local-data) into the database and API curve directory.",
    epilog="Example: quantagent backtest import local-data --database-url sqlite:////tmp/backtests.db --equity-dir /tmp/backtest-equity",
)
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path), default="local-data")
@click.option("--database-url", required=True, help="Destination SQLAlchemy database URL (with the existing schema).")
@click.option("--hipotesis-dir", type=click.Path(file_okay=False, path_type=Path), default=HIPOTESIS_DIR,
              show_default=True, help=f"Directory with hypothesis YAML files; default: {HIPOTESIS_DIR}.")
@click.option("--equity-dir", type=click.Path(file_okay=False, path_type=Path),
              default=lambda: os.environ.get("QUANTAGENT_BACKTEST_EQUITY_DIR", "local-data/backtest-equity"),
              help="Output directory for API curves; default: QUANTAGENT_BACKTEST_EQUITY_DIR or local-data/backtest-equity.")
@click.option("--dry-run", is_flag=True, help="Count new runs without writing database rows or curves (default: disabled).")
def import_command(root, database_url, hipotesis_dir, equity_dir, dry_run):
    engine = create_engine(database_url)
    imported = skipped = 0
    try:
        with Session(engine) as session:
            known_ids = known_hypothesis_ids(session, hipotesis_dir)
            existing = set(session.scalars(select(BacktestRun.data_source)).all())
            for path in sorted(root.rglob("equity/*.csv")):
                source = path.relative_to(root).as_posix()
                if source in existing:
                    skipped += 1
                    continue
                if dry_run:
                    imported += 1
                    continue
                parts = path.relative_to(root).parts[:-2]
                group = parts[-1] if parts else root.name
                try:
                    curve = pd.read_csv(path, usecols=["date", "equity"])
                    dates = pd.to_datetime(curve["date"], errors="raise")
                    values = pd.to_numeric(curve["equity"], errors="raise")
                    if (curve.empty or dates.isna().any() or not dates.is_monotonic_increasing
                            or dates.duplicated().any() or not values.map(math.isfinite).all()
                            or (values <= 0).any()):
                        raise ValueError("expected ordered unique dates and finite positive equity")
                    strategy, symbol = path.stem.rsplit("-", 1) if "-" in path.stem else (group, path.stem)
                except (ValueError, KeyError) as exc:
                    raise click.ClickException(f"{source}: {exc}") from exc

                stage = next((part for part in parts if part.startswith("etapa")), None)
                config = {"group": group, "stage": stage, "strategy": strategy, "source": source}
                if group.startswith("cartera-") or group == "rebalanceo-fin-de-mes":
                    config["hypothesis_name"] = group
                elif group.startswith("m5-"):
                    config["hypothesis_name"] = group.removeprefix("m5-")

                # Same Sharpe convention as the engine: observed periods/year, 2% rf, sample std.
                returns = values.pct_change().dropna()
                years = (dates.iloc[-1] - dates.iloc[0]).total_seconds() / (365.25 * 86400)
                deviation = returns.std()
                sharpe = None
                if len(returns) > 1 and years > 0 and deviation > 0:
                    periods = len(returns) / years
                    sharpe = float((returns.mean() - 0.02 / periods) / deviation * math.sqrt(periods))
                run = BacktestRun(
                    name=" · ".join((*parts, path.stem)), timeframe="1d", assets=[symbol],
                    start_date=dates.iloc[0].to_pydatetime(), end_date=dates.iloc[-1].to_pydatetime(),
                    data_source=source, config_snapshot=config, sharpe_ratio=sharpe,
                    hypothesis_id=resolve_hypothesis_id(config.get("hypothesis_name"), known_ids),
                    max_drawdown=float(((values.cummax() - values) / values.cummax()).max()),
                    total_pnl=float(values.iloc[-1] - values.iloc[0]),
                )
                trades_path = path.parent.parent / "trades" / path.name
                if trades_path.is_file():
                    trades = pd.read_csv(trades_path)
                    if {"symbol", "exit_time", "pnl"} <= set(trades.columns):
                        closed = trades[trades["exit_time"].notna() & trades["pnl"].notna()]
                        if trades["symbol"].eq(symbol).all():
                            run.total_trades = len(closed)
                            run.win_rate = float(closed["pnl"].gt(0).mean()) if len(closed) else None
                session.add(run)
                session.flush()
                equity_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"timestamp": dates.map(lambda date: date.isoformat()), "equity": values}).to_csv(
                    equity_dir / f"{run.id}.csv", index=False,
                )
                # Commit each complete run so a retry can resume after an interrupted import.
                session.commit()
                existing.add(source)
                imported += 1
    finally:
        engine.dispose()
    click.echo(f"{'Importaría' if dry_run else 'Importadas'}: {imported}; existentes: {skipped}")
