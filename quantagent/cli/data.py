"""CLI de datos: snapshots diarios congelados desde Yahoo (iuf D4-D6, V02)."""

from __future__ import annotations

from datetime import date
from typing import Dict

import click
import pandas as pd

from quantagent.data.snapshot import (
    SnapshotError,
    check_snapshot,
    load_manifest,
    snapshot_root,
    verify_snapshot,
    write_snapshot,
)

_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def download_daily(symbol: str, start: str) -> pd.DataFrame:
    """Baja velas diarias de Yahoo con `auto_adjust=False`: OHLC y volumen ajustados por splits, `Adj Close` aparte."""
    import yfinance as yf

    raw = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=False)
    if raw.empty:
        return raw
    df = raw[list(_COLUMNS)].rename(columns=_COLUMNS)
    df.index = pd.DatetimeIndex(raw.index.tz_localize(None).normalize(), name="timestamp")
    return df


@click.group("data", help="Datos de mercado congelados.")
def data_group() -> None:
    """Root command group for data operations."""


@data_group.group("snapshot", help="Snapshots en Parquet fuera del repo ($QUANTAGENT_SNAPSHOT_DIR).")
def snapshot_group() -> None:
    """Snapshot subcommands."""


@snapshot_group.command("create", help="Baja los símbolos de Yahoo (ajustados por splits, sin ajuste por dividendos) y congela el snapshot.")
@click.option("--name", required=True, help="Nombre del snapshot, por ejemplo etf-1d-2026-10.")
@click.option("--symbols", required=True, help="Símbolos separados por coma.")
@click.option("--start", required=True, help="Primera fecha, YYYY-MM-DD.")
@click.option("--end", required=True, help="Última sesión incluida, YYYY-MM-DD. Debe ser anterior a hoy (la vela del día en curso no está cerrada).")
def create_snapshot(name: str, symbols: str, start: str, end: str) -> None:
    try:
        last = date.fromisoformat(end)
    except ValueError:
        raise click.ClickException(f"--end '{end}' no es una fecha YYYY-MM-DD")
    if last >= date.today():
        raise click.ClickException(f"--end {end} es hoy o futuro: la vela del día en curso no está cerrada; usá una fecha anterior a {date.today()}")
    try:
        if (snapshot_root() / name).exists():
            raise SnapshotError(f"el snapshot '{name}' ya existe: se congela y no se vuelve a bajar")
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in [s.strip().upper() for s in symbols.split(",") if s.strip()]:
            frame = download_daily(symbol, start)
            frames[symbol] = frame[frame.index <= pd.Timestamp(last)]
            if frames[symbol].empty:
                raise SnapshotError(f"{symbol}: Yahoo no devolvió filas entre {start} y {end}")
        manifest = write_snapshot(name, frames)
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    rows = sum(info["rows"] for info in manifest["symbols"].values())
    click.echo(f"Snapshot {name} creado: {len(frames)} símbolos, {rows} filas")


@snapshot_group.command("verify", help="Recalcula los hashes contra el manifiesto.")
@click.option("--name", required=True)
def verify_snapshot_cmd(name: str) -> None:
    try:
        verify_snapshot(name)
        symbols = load_manifest(name)["symbols"]
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"OK {len(symbols)} símbolos, {sum(i['rows'] for i in symbols.values())} filas")


@snapshot_group.command("check", help="Reporte de calidad contra el calendario NYSE. Exit 1 solo si faltan sesiones.")
@click.option("--name", required=True)
def check_snapshot_cmd(name: str) -> None:
    try:
        report = check_snapshot(name)
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    for symbol, r in report.items():
        click.echo(
            f"{symbol}: faltan {len(r['faltan'])}, de más {len(r['de_mas'])}, OHLC incoherente {r['ohlc_incoherente']}, "
            f"volumen 0 {r['volumen_cero']}, saltos >20% {len(r['saltos'])}"
        )
        for label, key in (("faltan", "faltan"), ("de más", "de_mas"), ("saltos", "saltos")):
            if r[key]:
                click.echo(f"  {label}: {', '.join(r[key][:10])}{' ...' if len(r[key]) > 10 else ''}")
    if any(r["faltan"] for r in report.values()):
        raise SystemExit(1)
