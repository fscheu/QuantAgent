"""CLI de datos: snapshots diarios congelados desde Yahoo (iuf D4-D6, V02)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import click
import pandas as pd

from quantagent.data.acciones import crear_snapshot_acciones, reporte_cobertura
from quantagent.data.fundamentales import (
    crear_snapshot_fundamentales,
    reporte_cobertura_fundamentales,
)
from quantagent.data.macro import crear_snapshot_macro, verificar_snapshot_macro
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
@click.argument("name_arg", required=False, default=None)
@click.option("--name", default=None)
def verify_snapshot_cmd(name_arg: Optional[str], name: Optional[str]) -> None:
    snapshot_name = name or name_arg
    if not snapshot_name:
        raise click.ClickException("Falta el nombre del snapshot (--name o argumento)")
    try:
        verify_snapshot(snapshot_name)
        symbols = load_manifest(snapshot_name)["symbols"]
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


@data_group.group("macro", help="Snapshots macro de ALFRED (todas las versiones) en Parquet.")
def macro_group() -> None:
    """Macro snapshot subcommands."""


@macro_group.command("create", help="Baja todas las versiones de las series de ALFRED y congela el snapshot (requiere FRED_API_KEY).")
@click.option("--name", required=True, help="Nombre del snapshot, por ejemplo macro-2026-10.")
@click.option("--series", required=True, help="Series de FRED separadas por coma, por ejemplo UNRATE,CPIAUCSL.")
@click.option("--nota", "notas_raw", multiple=True, help="Nota sobre una serie en formato SERIE=texto.")
def create_macro(name: str, series: str, notas_raw: tuple[str, ...]) -> None:
    from quantagent.data.sources.fred import FredClient

    notas = {}
    for item in notas_raw:
        if "=" not in item:
            raise click.ClickException(f"formato inválido en --nota: '{item}', debe ser SERIE=texto")
        serie, texto = item.split("=", 1)
        serie = serie.strip().upper()
        if not serie:
            raise click.ClickException(f"formato inválido en --nota: '{item}', falta la serie")
        notas[serie] = texto

    try:
        manifest = crear_snapshot_macro(
            name,
            [s.strip().upper() for s in series.split(",") if s.strip()],
            FredClient(),
            notas=notas or None,
        )
    except (SnapshotError, RuntimeError) as exc:
        raise click.ClickException(str(exc))
    rows = sum(info["rows"] for info in manifest["series"].values())
    click.echo(f"Snapshot macro {name} creado: {len(manifest['series'])} series, {rows} filas")


@macro_group.command("verify", help="Recalcula los hashes contra el manifiesto.")
@click.argument("name_arg", required=False, default=None)
@click.option("--name", default=None)
def verify_macro(name_arg: Optional[str], name: Optional[str]) -> None:
    snapshot_name = name or name_arg
    if not snapshot_name:
        raise click.ClickException("Falta el nombre del snapshot macro (--name o argumento)")
    try:
        series = verificar_snapshot_macro(snapshot_name)["series"]
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"OK {len(series)} series, {sum(i['rows'] for i in series.values())} filas")


@data_group.group("consenso", help="Consenso diario de estimaciones (EPS, ingresos).")
def consenso_group() -> None:
    """Consenso subcommands."""


@consenso_group.command("foto", help="Guarda la foto diaria de consenso de EPS en Parquet.")
@click.option("--simbolos", default=None, help="Símbolos separados por coma, ej. AAPL,MSFT,NVDA.")
@click.option("--archivo", default=None, type=click.Path(exists=True, dir_okay=False), help="Archivo con un símbolo por línea.")
@click.option("--fecha", default=None, help="Fecha de la foto en YYYY-MM-DD (por defecto hoy).")
@click.option("--fuente", default="yahoo", type=click.Choice(["yahoo", "fmp"], case_sensitive=False), help="Fuente de datos (yahoo o fmp).")
def foto_consenso_cmd(simbolos: Optional[str], archivo: Optional[str], fecha: Optional[str], fuente: str = "yahoo") -> None:
    from quantagent.data.consenso import guardar_foto

    simbolos_lista = []
    if simbolos:
        simbolos_lista.extend([s.strip().upper() for s in simbolos.split(",") if s.strip()])
    if archivo:
        with open(archivo, "r", encoding="utf-8") as f:
            for linea in f:
                s = linea.strip().upper()
                if s and not s.startswith("#"):
                    simbolos_lista.append(s)

    simbolos_unicos = list(dict.fromkeys(simbolos_lista))
    if not simbolos_unicos:
        raise click.ClickException("Se debe indicar al menos un símbolo mediante --simbolos o --archivo")

    fecha_obj = None
    if fecha:
        try:
            fecha_obj = date.fromisoformat(fecha)
        except ValueError:
            raise click.ClickException(f"--fecha '{fecha}' no es una fecha YYYY-MM-DD válida")

    try:
        if fuente.lower() == "yahoo":
            from quantagent.data.sources.yahoo_eps import YahooEpsClient

            cliente = YahooEpsClient()
            carpeta = "consenso-eps-yahoo"
        else:
            from quantagent.data.sources.fmp import FmpClient

            cliente = FmpClient()
            carpeta = "consenso-eps"

        destino = guardar_foto(simbolos_unicos, cliente=cliente, fecha=fecha_obj, carpeta=carpeta)
        click.echo(f"Foto de consenso guardada en {destino}")
    except (FileExistsError, RuntimeError, SnapshotError) as exc:
        raise click.ClickException(str(exc))


@data_group.group("acciones", help="Snapshot de acciones del S&P 500 2007-2018 (Yahoo o Quandl WIKI por ticker).")
def acciones_group() -> None:
    """Acciones snapshot subcommands."""


@acciones_group.command("create", help="Arma el snapshot: Yahoo si cubre la pertenencia al índice, si no WIKI_PRICES.csv.")
@click.option("--name", required=True, help="Nombre del snapshot, por ejemplo acciones-sp500-2026-10.")
@click.option("--wiki-csv", default="/tmp/wiki/WIKI_PRICES.csv", type=click.Path(exists=True, dir_okay=False), show_default=True)
def create_acciones(name: str, wiki_csv: str) -> None:
    try:
        manifest = crear_snapshot_acciones(name, Path(wiki_csv))
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    fuentes = pd.Series(manifest["fuentes"]).value_counts().to_dict()
    click.echo(f"Snapshot {name} creado: {len(manifest['symbols'])} símbolos, fuentes {fuentes}")


@acciones_group.command("cobertura", help="Reporte de cobertura de precios sobre los miembro-días 2007-2018; lo guarda en la carpeta del snapshot.")
@click.option("--name", required=True)
def cobertura_acciones(name: str) -> None:
    try:
        reporte = reporte_cobertura(name)
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    carpeta = snapshot_root() / name
    (carpeta / "cobertura.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(reporte["por_ticker"]).to_csv(carpeta / "cobertura.csv", index=False)
    click.echo(f"Miembro-días con precio: {reporte['pct_global']:.2f}% ({reporte['miembro_dias_con_precio']}/{reporte['miembro_dias']})")
    click.echo("Por año: " + ", ".join(f"{a}: {p:.2f}%" for a, p in reporte["pct_por_anio"].items()))
    click.echo(f"Sin precio ({len(reporte['sin_precio'])}): {', '.join(reporte['sin_precio'])}")
    click.echo(f"Terminan antes de su salida ({len(reporte['terminan_antes'])}): {reporte['terminan_antes']}")
    click.echo(f"Saltos >50% ({len(reporte['saltos'])}): {reporte['saltos']}")
    if reporte.get("reuso_cambiados_a_wiki"):
        click.echo(f"Reúso cambiados a WIKI ({len(reporte['reuso_cambiados_a_wiki'])}): {reporte['reuso_cambiados_a_wiki']}")
    if reporte.get("yahoo_sin_wiki"):
        click.echo(f"Yahoo sin WIKI ({len(reporte['yahoo_sin_wiki'])}): {list(reporte['yahoo_sin_wiki'].keys())}")


@data_group.group("fundamentales", help="Snapshot de fundamentales de SEC EDGAR (XBRL).")
def fundamentales_group() -> None:
    """Fundamentales snapshot subcommands."""


@fundamentales_group.command("create", help="Descarga companyfacts desde SEC EDGAR y crea el snapshot.")
@click.option("--name", required=True, help="Nombre del snapshot, por ejemplo fundamentales-sp500-2026-10.")
@click.option("--tickers-de", default="sp500-2026-10", help="Snapshot de composición del que extraer miembros 2007..hoy.")
@click.option("--tickers", default=None, help="Lista explícita de tickers separados por coma.")
def create_fundamentales(name: str, tickers_de: str, tickers: Optional[str]) -> None:
    if tickers:
        lista_tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        df_comp = pd.read_parquet(snapshot_root() / tickers_de / load_manifest(tickers_de)["files"]["composicion"]["file"])
        lista_tickers = sorted({t.strip().upper() for s in df_comp[df_comp["date"].astype(str) >= "2007-01-01"]["tickers"] for t in str(s).split(",") if t.strip()})
    try:
        manifest = crear_snapshot_fundamentales(name, lista_tickers)
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Snapshot {name} creado: {len(manifest['symbols'])} tickers con datos")


@fundamentales_group.command("verify", help="Recalcula los hashes contra el manifiesto.")
@click.argument("name_arg", required=False, default=None)
@click.option("--name", default=None)
def verify_fundamentales(name_arg: Optional[str], name: Optional[str]) -> None:
    snapshot_name = name or name_arg
    if not snapshot_name:
        raise click.ClickException("Falta el nombre del snapshot (--name o argumento)")
    try:
        verify_snapshot(snapshot_name)
        symbols = load_manifest(snapshot_name)["symbols"]
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"OK {len(symbols)} tickers, {sum(i['rows'] for i in symbols.values())} filas")


@fundamentales_group.command("cobertura", help="Reporte de cobertura por métrica desde 2010 y tickers sin CIK.")
@click.option("--name", required=True)
@click.option("--composicion", default="sp500-2026-10")
def cobertura_fundamentales_cmd(name: str, composicion: str) -> None:
    try:
        rep = reporte_cobertura_fundamentales(name, composicion=composicion)
    except SnapshotError as exc:
        raise click.ClickException(str(exc))
    carpeta = snapshot_root() / name
    (carpeta / "cobertura.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    click.echo("Cobertura miembro-día desde 2010: " + ", ".join(f"{m}: {info['pct']:.2f}%" for m, info in rep["cobertura"].items()))
    click.echo(f"Sin CIK ({rep['sin_cik']['cantidad']}): {', '.join(rep['sin_cik']['ejemplos'])}")
    if rep.get("muestra"):
        click.echo(f"Muestra: {rep['muestra']}")
