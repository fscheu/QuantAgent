"""Inspección e invalidación de corridas con backup del registro."""

import csv
import json
from datetime import datetime, timedelta, timezone

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


@click.group("intentos", help="Manage the run attempts log.")
def intentos_group():
    pass


@intentos_group.command("epoca-limpia", help="Record the start of the clean period.")
@click.option("--desde", required=True, help="Start timestamp in ISO 8601 UTC format.")
@click.option("--motivo", required=True, help="Reason for recording the clean period.")
def epoca_limpia(desde, motivo):
    desde = _fecha_utc(desde)
    path = snapshot_root() / "epoca_limpia.json"
    marca = {
        "desde_utc": desde.isoformat(),
        "motivo": motivo,
        "creada_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(marca, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    click.echo(f"Época limpia guardada: {path}")


@intentos_group.command("invalidar", help="List matching rows; invalidate them with a backup only when --aplicar is set.")
@click.option("--hipotesis", required=True, help="Hypothesis ID whose attempts should be selected.")
@click.option("--desde", help="Inclusive start timestamp in ISO 8601 UTC format.")
@click.option("--hasta", help="Inclusive end timestamp in ISO 8601 UTC format.")
@click.option("--motivo", required=True, help="Reason for invalidating the selected attempts.")
@click.option("--aplicar", is_flag=True, help="Write invalidation marks and create a backup; otherwise show a dry run.")
def invalidar(hipotesis, desde, hasta, motivo, aplicar):
    desde = _fecha_utc(desde) if desde is not None else None
    hasta = _fecha_utc(hasta) if hasta is not None else None
    path = snapshot_root() / "intentos.csv"
    filas = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as archivo:
            reader = csv.DictReader(archivo)
            campos = reader.fieldnames
            filas = list(reader)
    if not any(fila["hipotesis"] == hipotesis for fila in filas):
        raise click.ClickException(f"Hipótesis inexistente en intentos.csv: {hipotesis}")
    seleccionadas = []
    for fila in filas:
        if fila["hipotesis"] != hipotesis:
            continue
        fecha = _fecha_utc(fila["fecha_utc"])
        if (desde is None or fecha >= desde) and (hasta is None or fecha <= hasta):
            seleccionadas.append(fila)
    if aplicar:
        marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup = path.with_name(f"{path.name}.bak-{marca}")
        with backup.open("xb") as archivo:
            archivo.write(path.read_bytes())
        if "estado" not in campos:
            campos.append("estado")
        for fila in seleccionadas:
            fila["estado"] = "invalida_bug"
        with path.open("w", newline="", encoding="utf-8") as archivo:
            writer = csv.DictWriter(archivo, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas)
        click.echo(f"Aplicado: {len(seleccionadas)} filas; motivo: {motivo}; backup: {backup}")
    else:
        click.echo(f"Dry-run: {len(seleccionadas)} filas; motivo: {motivo}")
    for fila in seleccionadas:
        click.echo(json.dumps(fila, ensure_ascii=False))
