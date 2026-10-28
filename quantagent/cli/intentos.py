"""Inspección de corridas que se invalidarían, sin modificar el registro."""

import csv
import json
from datetime import datetime, timedelta

import click

from quantagent.data.snapshot import snapshot_root


def _fecha_utc(texto):
    try:
        fecha = datetime.fromisoformat(texto.replace("Z", "+00:00"))
        if fecha.utcoffset() != timedelta(0):
            raise ValueError
        return fecha
    except ValueError as exc:
        raise click.ClickException(f"Fecha inválida: {texto!r}; se requiere ISO con zona UTC") from exc


@click.group("intentos", help="Administrar el registro de intentos.")
def intentos_group():
    pass


@intentos_group.command("invalidar", help="Listar las filas que se invalidarían (dry-run).")
@click.option("--hipotesis", required=True)
@click.option("--desde")
@click.option("--hasta")
@click.option("--motivo", required=True)
def invalidar(hipotesis, desde, hasta, motivo):
    desde = _fecha_utc(desde) if desde is not None else None
    hasta = _fecha_utc(hasta) if hasta is not None else None
    path = snapshot_root() / "intentos.csv"
    filas = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as archivo:
            filas = [fila for fila in csv.DictReader(archivo) if fila["hipotesis"] == hipotesis]
    if not filas:
        raise click.ClickException(f"Hipótesis inexistente en intentos.csv: {hipotesis}")
    seleccionadas = []
    for fila in filas:
        fecha = _fecha_utc(fila["fecha_utc"])
        if (desde is None or fecha >= desde) and (hasta is None or fecha <= hasta):
            seleccionadas.append(fila)
    click.echo(f"Dry-run: {len(seleccionadas)} filas; motivo: {motivo}")
    for fila in seleccionadas:
        click.echo(json.dumps(fila, ensure_ascii=False))
