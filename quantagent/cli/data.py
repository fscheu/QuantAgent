"""CLI de datos: snapshots diarios congelados desde Yahoo (iuf D4-D6, V02)."""

from __future__ import annotations

from typing import Dict

import click
import pandas as pd

from quantagent.data.snapshot import SnapshotError, load_manifest, snapshot_root, verify_snapshot, write_snapshot

_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def download_daily(symbol: str, start: str) -> pd.DataFrame:
    """Baja velas diarias de Yahoo sin ajustar: OHLC, `Adj Close` y volumen tal cual."""
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


@snapshot_group.command("create", help="Baja los símbolos de Yahoo sin ajustar y congela el snapshot.")
@click.option("--name", required=True, help="Nombre del snapshot, por ejemplo etf-1d-2026-10.")
@click.option("--symbols", required=True, help="Símbolos separados por coma.")
@click.option("--start", required=True, help="Primera fecha, YYYY-MM-DD.")
def create_snapshot(name: str, symbols: str, start: str) -> None:
    try:
        if (snapshot_root() / name).exists():
            raise SnapshotError(f"el snapshot '{name}' ya existe: se congela y no se vuelve a bajar")
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in [s.strip().upper() for s in symbols.split(",") if s.strip()]:
            frames[symbol] = download_daily(symbol, start)
            if frames[symbol].empty:
                raise SnapshotError(f"{symbol}: Yahoo no devolvió filas desde {start}")
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
