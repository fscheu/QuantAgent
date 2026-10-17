"""CLI de admisión de hipótesis (QuantAgent-d4cq.14.1)."""

import csv
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from math import isnan
from pathlib import Path

import click
import pandas as pd
from tabulate import tabulate

from quantagent.data.snapshot import SnapshotError, read_snapshot
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.candado import CandadoError, abrir, fecha_apertura
from quantagent.fabrica.cartera import evaluar_cartera
from quantagent.fabrica.estado import registrar_estado, resumen_tanda, ultimo_estado
from quantagent.fabrica.filtro import filtrar_activo, parametros_elegidos, veredicto_etapa1
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.informe import datos_informe, render_informe
from quantagent.fabrica.validacion import alarma_filtro_motor, validar_etapa2, validar_etapa3
from quantagent.settings import COSTOS_PERFILES

from .sweep import registrar_intento


@click.group("hipotesis", help="Evaluar hipótesis de la fábrica de estrategias.")
def hipotesis_group():
    pass


@hipotesis_group.command("tanda", help="Procesar los YAML en orden y guardar informes, logs y resumen.")
@click.argument("carpeta_yaml", type=click.Path(exists=True, file_okay=False))
@click.option("--snapshot", "snapshot_name", required=True)
@click.option("--carpeta", type=click.Path(file_okay=False), required=True)
@click.pass_context
def tanda_command(ctx, carpeta_yaml, snapshot_name, carpeta):
    desde = datetime.now(timezone.utc)
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)
    for archivo in sorted(Path(carpeta_yaml).glob("*.yaml")):
        try:
            h = cargar_hipotesis(archivo)
        except Exception:
            (destino / f"{archivo.stem}.log").write_text(traceback.format_exc(), encoding="utf-8")
            continue
        salida = destino / h.id
        salida.mkdir(parents=True, exist_ok=True)
        with (salida / "corrida.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    estado = ultimo_estado(h.id)
                    if estado and (estado["veredicto"] in ("CORTA", "ALARMA") or
                                   (estado["veredicto"] == "PASA" and estado["etapa"] == "3")):
                        click.echo(f"ya evaluada: etapa {estado['etapa']} {estado['veredicto']}; se saltea")
                    else:
                        ctx.invoke(correr_command, archivo=str(archivo), snapshot_name=snapshot_name, carpeta=str(salida))
                except click.exceptions.Exit:
                    pass
                except Exception:
                    traceback.print_exc()
                try:
                    ctx.invoke(informe_command, hipotesis_id=h.id, out=str(salida / "informe.html"), archivo=str(archivo))
                except Exception:
                    traceback.print_exc()
    (destino / "resumen.md").write_text(resumen_tanda(desde), encoding="utf-8")
    click.echo(f"Tanda terminada: {destino / 'resumen.md'}")


@hipotesis_group.command("resumen", help="Resumir los últimos estados y los intentos de la tanda.")
@click.option("--desde", type=click.DateTime(["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z"]),
              help="Inicio UTC inclusivo (fecha o fecha y hora con zona).")
@click.option("--out", type=click.Path(dir_okay=False))
def resumen_command(desde, out):
    if desde is not None:
        desde = desde.replace(tzinfo=timezone.utc) if desde.tzinfo is None else desde
    texto = resumen_tanda(desde)
    if out:
        Path(out).write_text(texto)
    click.echo(texto, nl=False)


def admitir_hipotesis(archivo, snapshot_name, prueba_unica=False):
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
    if prueba_unica:
        fecha = fecha_apertura(h.id, "prueba")
        if fecha is not None:
            click.echo(f"prueba de {h.id} ya abierta el {fecha}; no se vuelve a correr")
            raise click.exceptions.Exit(1)
    if not ok:
        registrar_intento(
            h.clase, h.activos[0], snapshot_name, "etf", 1.0, h.parametros,
            {"desde": "2007-01-01", "hasta": "2018-12-31", "trades": 0, "pnl": "", "sharpe": ""},
            hipotesis=h.id, etapa="0", sharpe_base="",
        )
    return h, ok, motivos


def filtrar_hipotesis(h, snapshot_name):
    """Filtrar y registrar intento de cartera; devolver métricas, parámetros y veredicto."""
    resultados, parametros = [], {}
    retornos_dict, cierres_dict = {}, {}
    filas_tabla = []
    total_trades = 0
    piernas_map = {}
    for i, p in enumerate(h.piernas):
        piernas_map[i] = list(p.activos)
        for activo in p.activos:
            try:
                velas_raw = read_snapshot(snapshot_name, activo).loc["2007-01-01":"2018-12-31"]
                if velas_raw.empty:
                    raise ValueError("Sin velas entre 2007 y 2018")
                cierres_dict[activo] = velas_raw["close"]
                velas = velas_raw.rename_axis("timestamp").reset_index()
                r = filtrar_activo(p, velas, costos=COSTOS_PERFILES["etf"], capital=100_000.0)
            except (ValueError, SnapshotError) as exc:
                raise click.ClickException(str(exc)) from exc
            resultados.append([activo, r["sharpe_tasa_0"], r["trades"]])
            filas_tabla.append([p.clase, activo, r["sharpe_tasa_0"], r["trades"]] if len(h.piernas) > 1 else [activo, r["sharpe_tasa_0"], r["trades"]])
            retornos_dict[(i, activo)] = r["retornos"]
            total_trades += r["trades"]
            parametros[activo] = parametros_elegidos(r["ganadores"])
    cierres = pd.DataFrame(cierres_dict)
    sharpe_car = evaluar_cartera(cierres, piernas_map, retornos_dict)
    veredicto, motivo = veredicto_etapa1(sharpe_car, total_trades)
    combo = h.parametros if len(h.piernas) == 1 else {p.clase: p.parametros for p in h.piernas}
    estrategia = h.clase if len(h.piernas) == 1 else "+".join(p.clase for p in h.piernas)
    sharpe_csv = sharpe_car if (sharpe_car is not None and not (isinstance(sharpe_car, float) and isnan(sharpe_car))) else ""
    registrar_intento(
        estrategia, "cartera", snapshot_name, "etf", 1.0, combo,
        {"desde": "2007-01-01", "hasta": "2018-12-31", "trades": total_trades,
         "pnl": "", "sharpe": sharpe_csv}, hipotesis=h.id, etapa="1",
        sharpe_base="tasa_0",
    )
    return resultados, parametros, veredicto, motivo, sharpe_car, total_trades, filas_tabla


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
    ret = filtrar_hipotesis(h, snapshot_name)
    resultados, parametros, veredicto, motivo = ret[:4]
    sharpe_car = ret[4] if len(ret) > 4 else None
    total_trades = ret[5] if len(ret) > 5 else sum(r[2] for r in resultados)
    filas_tabla = ret[6] if len(ret) > 6 else (
        [[p.clase, *r] for p in h.piernas for r in resultados] if len(h.piernas) > 1 else resultados
    )
    headers = ["pierna", "activo", "sharpe_fuera_muestra", "trades"] if len(h.piernas) > 1 else ["activo", "sharpe_fuera_muestra", "trades"]
    fila_cartera = ["cartera", "", sharpe_car, total_trades] if len(h.piernas) > 1 else ["cartera", sharpe_car, total_trades]
    click.echo(tabulate([*filas_tabla, fila_cartera], headers=headers, tablefmt="github"))
    if out:
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["activo", "sharpe_fuera_muestra", "trades", "params", "veredicto"])
            writer.writerows(
                [*fila, json.dumps(parametros[fila[0]], sort_keys=True), veredicto] for fila in resultados
            )
    click.echo(f"{veredicto} ({motivo})")


@hipotesis_group.command("correr", help="Ejecutar las etapas 0 a 3, con corte temprano y estado.")
@click.argument("archivo", type=click.Path(exists=True, dir_okay=False))
@click.option("--snapshot", "snapshot_name", required=True)
@click.option("--carpeta", type=click.Path(file_okay=False), required=True)
def correr_command(archivo, snapshot_name, carpeta):
    h, ok, motivos = admitir_hipotesis(archivo, snapshot_name, True)
    etapa, veredicto = "0", "PASA" if ok else "CORTA"
    motivo = "; ".join(motivos) if motivos else "admisión aceptada"
    registrar_estado(h.id, etapa, veredicto, motivo)
    click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
    if ok:
        resultados, parametros, veredicto, motivo = filtrar_hipotesis(h, snapshot_name)[:4]
        etapa = "1"
        registrar_estado(h.id, etapa, veredicto, motivo)
        click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
        if veredicto == "PASA":
            try:
                veredicto, filas = validar_etapa2(h, snapshot_name, parametros, carpeta)
            except Exception as exc:
                registrar_estado(h.id, "2", "ERROR", f"{type(exc).__name__}: {exc}"[:200])
                raise
            etapa = "2"
            motivos = [m for fila in filas for m in fila["motivos"]]
            motivo = "; ".join(motivos) if motivos else "control coincide y al menos un activo pasa V30"
            if veredicto == "PASA":
                activos = ", ".join(f["activo"] for f in filas if f["v30"] == "PASA")
                motivo = f"pasan V30: {activos}; control coincide en {len(filas)} de {len(filas)}"
                alarma, activos = alarma_filtro_motor(resultados, filas)
                if alarma:
                    veredicto = "ALARMA"
                    motivo = "filtro optimista en " + ", ".join(activos)
            registrar_estado(h.id, etapa, veredicto, motivo)
            click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
            if veredicto == "PASA":
                etapa = "3"
                try:
                    abrir(h.id, "prueba")
                except CandadoError as exc:
                    veredicto, motivo = "CORTA", str(exc)
                else:
                    try:
                        veredicto, filas = validar_etapa3(h, snapshot_name, parametros, Path(carpeta) / "etapa3")
                    except Exception as exc:
                        registrar_estado(h.id, "3", "ERROR", f"{type(exc).__name__}: {exc}"[:200])
                        raise
                    click.echo(tabulate([[f["activo"], f["v30"], f["dsr"]] for f in filas],
                                        headers=["activo", "V30", "DSR"], tablefmt="github"))
                    motivo = "; ".join(m for f in filas for m in f["motivos"])
                    if veredicto == "PASA":
                        motivo = "pasan V30 y DSR: " + ", ".join(f["activo"] for f in filas if not f["motivos"])
                registrar_estado(h.id, etapa, veredicto, motivo)
                click.echo(f"Etapa {etapa}: {veredicto} ({motivo})")
    click.echo(f"Llegó a la etapa {etapa}: {veredicto} ({motivo})")
    raise click.exceptions.Exit(int(veredicto != "PASA"))


@hipotesis_group.command("informe", help="Página HTML con el veredicto y los motivos de una hipótesis.")
@click.argument("hipotesis_id")
@click.option("--out", type=click.Path(dir_okay=False), required=True)
@click.option("--archivo", type=click.Path(exists=True, dir_okay=False), help="YAML de la hipótesis.")
def informe_command(hipotesis_id, out, archivo):
    try:
        h = cargar_hipotesis(archivo) if archivo else None
        html = render_informe(datos_informe(hipotesis_id), h)
    except (ValueError, SnapshotError) as exc:
        raise click.ClickException(str(exc)) from exc
    Path(out).write_text(html, encoding="utf-8")
    click.echo(f"Informe en {out}")
