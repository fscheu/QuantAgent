"""Commands for the daily market bar store."""

import os
from datetime import date

import click
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.data.market_bars import DailyAlpacaClient, MARKET_BAR_SYMBOLS, update_market_bars


@click.group("mercado", help="Manage stored market data.")
def market_group():
    pass


@market_group.group("barras", help="Update stored daily market bars.")
def bars_group():
    pass


@bars_group.command("actualizar", help="Update daily bars from Alpaca or a frozen snapshot.",
                    epilog="Example: quantagent mercado barras actualizar --simbolos SPY,QQQ --desde 2024-01-01")
@click.option("--simbolos", default=None, help="Comma-separated symbols; default: the 31-symbol universe.")
@click.option("--desde", default=None, help="First session date in YYYY-MM-DD format; default: first missing session.")
@click.option("--base", default=None, help="SQLAlchemy database URL; default: QUANTAGENT_DATABASE_URL.")
@click.option("--snapshot", default=None, help="Snapshot name to backfill instead of downloading from Alpaca.")
@click.option("--dry-run", is_flag=True, help="Report changes without saving them; default: save changes.")
@click.option("--fallar-si-stale", is_flag=True, help="Exit with an error if any symbol is stale or has missing sessions; default: succeed.")
def update_bars(simbolos, desde, base, snapshot, dry_run, fallar_si_stale):
    symbols = [s.strip().upper() for s in simbolos.split(",") if s.strip()] if simbolos else list(MARKET_BAR_SYMBOLS)
    start = date.fromisoformat(desde) if desde else None
    engine = create_engine(base or os.environ["QUANTAGENT_DATABASE_URL"])
    with Session(engine) as session:
        result = update_market_bars(session, symbols, DailyAlpacaClient(), start=start,
                                    dry_run=dry_run, snapshot=snapshot)
        if dry_run:
            session.rollback()
    for symbol, item in result.items():
        session_date, stale, missing = item["status"]
        click.echo(f"{symbol}: {item['rows']} barras; session_date={session_date}; stale={stale}; missing_sessions={len(missing)}")
    if fallar_si_stale and any(stale or missing for _, stale, missing in (item["status"] for item in result.values())):
        raise click.ClickException("Uno o más símbolos tienen datos stale o sesiones faltantes.")
