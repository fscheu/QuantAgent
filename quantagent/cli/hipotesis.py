"""CLI de admisión de hipótesis (QuantAgent-d4cq.14.1)."""

import csv
import json
from pathlib import Path

import click
from tabulate import tabulate

from quantagent.data.snapshot import SnapshotError, read_snapshot
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.candado import CandadoError, abrir
from quantagent.fabrica.estado import registrar_estado
from quantagent.fabrica.filtro import filtrar_activo, parametros_elegidos, veredicto_etapa1
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.validacion import validar_etapa2, validar_etapa3
from quantagent.settings import COSTOS_PERFILES

from .sweep import registrar_intento


@click.group("hipotesis", help="Evaluar hipótesis de la fábrica de estrategias.")
def hipotesis_group():
    pass


def admitir_hipotesis(archivo, snapshot_name):
    """Admitir y registrar el rechazo, devolviendo hipótesis y motivos."""
    try:
        h = cargar_hipotesis(archivo)
        velas = read_snapshot(snapshot_name, h.activos[0]).loc["2007-01-01":"2018-12-31"]
        if velas.empty:
            raise ValueError("Sin velas entre 2007 y 2018")
        velas = velas.rename_axis("timestamp").reset_index().set_index("timestamp", drop=False)
        ok, motivos = admitir(h, velas)
    except (ValueError, SnapshotError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not ok:
        registrar_intento(
            h.clase, h.activos[0], snapshot_name, "etf", 1.0, h.parametros,
            {"desde": "2007-01-01", "hasta": "2018-12-31", "trades": 0, "pnl": "", "sharpe": ""},
            hipotesis=h.id, etapa="0",
        )
    return h, ok, motivos


def filtrar_hipotesis(h, snapshot_name):
    """Filtrar y registrar intentos; devolver métricas y parámetros por activo."""
    resultados, parametros = [], {}
    for activo in h.activos:
        try:
            velas = read_snapshot(snapshot_name, activo).loc["2007-01-01":"2018-12-31"]
            if velas.empty:
                raise ValueError("Sin velas entre 2007 y 2018")
            velas = velas.rename_axis("timestamp").reset_index()
            r = filtrar_activo(h, velas, costos=COSTOS_PERFILES["etf"], capital=100_000.0)
        except (ValueError, SnapshotError) as exc:
            raise click.ClickException(str(exc)) from exc
        resultados.append([activo, r["sharpe_tasa_0"], r["trades"]])
        parametros[activo] = parametros_elegidos(r["ganadores"])
        registrar_intento(
            h.clase, activo, snapshot_name, "etf", 1.0, h.parametros,
            {"desde": "2007-01-01", "hasta": "2018-12-31", "trades": r["trades"],
             "pnl": "", "sharpe": r["sharpe_tasa_0"]}, hipotesis=h.id, etapa="1",
        )
    veredicto, bajos = veredicto_etapa1(resultados)
    return resultados, parametros, veredicto, bajos


@hipotesis_group.command("filtrar", help="Cargar y admitir una hipótesis sobre 2007–2018.")
@click.argument("archivo", type=click.Path(exists=True, dir_okay=False))
@click.option("--snapshot", "snapshot_name", required=True)
@click.option("--out", type=click.Path(dir_okay=False), help="CSV de resultados y parámetros por activo.")
def filtrar_command(archivo, snapshot_name, out):
    h, ok, motivos = admitir_hipotesis(archivo, snapshot_name)
    if not ok:
        for motivo in motivos:
            click.echo(motivo)
        raise click.exceptions.Exit(1)
    resultados, parametros, veredicto, bajos = filtrar_hipotesis(h, snapshot_name)
    click.echo(tabulate(resultados, headers=["activo", "sharpe_fuera_muestra", "trades"], tablefmt="github"))
    if out:
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["activo", "sharpe_fuera_muestra", "trades", "params", "veredicto"])
            writer.writerows(
                [*fila, json.dumps(parametros[fila[0]], sort_keys=True), veredicto] for fila in resultados
            )
    click.echo(f"{veredicto} ({bajos} de {len(resultados)} bajo 0,10 o con menos de 30 trades)")


@hipotesis_group.command("correr", help="Ejecutar las etapas 0 a 3, con corte temprano y estado.")
@click.argument("archivo", type=click.Path(exists=True, dir_okay=False))
@click.option("--snapshot", "snapshot_name", required=True)
@click.option("--carpeta", type=click.Path(file_okay=False), required=True)
def correr_command(archivo, snapshot_name, carpeta):
    h, ok, motivos = admitir_hipotesis(archivo, snapshot_name)
    etapa, veredicto = "0", "PASA" if ok else "CORTA"
    motivo = "; ".join(motivos) if motivos else "admisión aceptada"
    registrar_estado(h.id, etapa, veredicto, motivo)
    click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
    if ok:
        resultados, parametros, veredicto, bajos = filtrar_hipotesis(h, snapshot_name)
        etapa = "1"
        motivo = f"{bajos} de {len(resultados)} bajo 0,10 o con menos de 30 trades"
        registrar_estado(h.id, etapa, veredicto, motivo)
        click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
        if veredicto == "PASA":
            veredicto, filas = validar_etapa2(h, snapshot_name, parametros, carpeta)
            etapa = "2"
            motivos = [m for fila in filas for m in fila["motivos"]]
            motivo = "; ".join(motivos) if motivos else "control coincide y al menos un activo pasa V30"
            if veredicto == "PASA":
                activos = ", ".join(f["activo"] for f in filas if f["v30"] == "PASA")
                motivo = f"pasan V30: {activos}; control coincide en {len(filas)} de {len(filas)}"
            registrar_estado(h.id, etapa, veredicto, motivo)
            click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
            if veredicto == "PASA":
                etapa = "3"
                try:
                    abrir(h.id, "prueba")
                except CandadoError as exc:
                    veredicto, motivo = "CORTA", str(exc)
                else:
                    veredicto, filas = validar_etapa3(h, snapshot_name, parametros, Path(carpeta) / "etapa3")
                    click.echo(tabulate([[f["activo"], f["v30"], f["dsr"]] for f in filas],
                                        headers=["activo", "V30", "DSR"], tablefmt="github"))
                    motivo = "; ".join(m for f in filas for m in f["motivos"])
                    if veredicto == "PASA":
                        motivo = "pasan V30 y DSR: " + ", ".join(f["activo"] for f in filas if not f["motivos"])
                registrar_estado(h.id, etapa, veredicto, motivo)
                click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
    click.echo(f"Llegó a la etapa {etapa}: {veredicto} ({motivo})")
    raise click.exceptions.Exit(int(veredicto == "CORTA"))
