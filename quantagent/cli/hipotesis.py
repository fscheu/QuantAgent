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

from quantagent.data.snapshot import SnapshotError, read_snapshot, snapshot_root
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.candado import CandadoError, abrir, fecha_apertura
from quantagent.fabrica.cartera import evaluar_cartera
from quantagent.fabrica.comparar import comparar
from quantagent.fabrica.estado import registrar_estado, resumen_tanda, ultimo_estado
from quantagent.fabrica.filtro import filtrar_activo, parametros_elegidos, veredicto_etapa1
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.informe import datos_informe, render_comparacion, render_informe
from quantagent.fabrica.validacion import (
    alarma_cartera,
    alarma_filtro_motor,
    validar_etapa2,
    validar_etapa3,
)
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
    resultados, parametros, retornos_dict, cierres_dict, filas_tabla = [], {}, {}, {}, []
    total_trades, piernas_map = 0, {}
    for i, p in enumerate(h.piernas):
        piernas_map[i] = list(p.activos)
        for activo in p.activos:
            try:
                v = read_snapshot(snapshot_name, activo).loc["2007-01-01":"2018-12-31"]
                if v.empty:
                    raise ValueError("Sin velas entre 2007 y 2018")
                cierres_dict[activo] = v["close"]
                r = filtrar_activo(p, v.rename_axis("timestamp").reset_index(), costos=COSTOS_PERFILES["etf"], capital=100_000.0)
            except (ValueError, SnapshotError) as exc:
                raise click.ClickException(str(exc)) from exc
            resultados.append([activo, r["sharpe_tasa_0"], r["trades"]])
            filas_tabla.append([p.clase, activo, r["sharpe_tasa_0"], r["trades"]] if len(h.piernas) > 1 else [activo, r["sharpe_tasa_0"], r["trades"]])
            retornos_dict[(i, activo)] = r["retornos"]
            total_trades += r["trades"]
            clave = (p.clase, activo) if len(h.piernas) > 1 else activo
            parametros[clave] = parametros_elegidos(r["ganadores"])
    sharpe_car = evaluar_cartera(pd.DataFrame(cierres_dict), piernas_map, retornos_dict)
    veredicto, motivo = veredicto_etapa1(sharpe_car, total_trades)
    combo = h.parametros if len(h.piernas) == 1 else {p.clase: p.parametros for p in h.piernas}
    estrategia = h.clase if len(h.piernas) == 1 else "+".join(p.clase for p in h.piernas)
    sharpe_csv = "" if sharpe_car is None or (isinstance(sharpe_car, float) and isnan(sharpe_car)) else sharpe_car
    registrar_intento(
        estrategia, "cartera", snapshot_name, "etf", 1.0, combo,
        {"desde": "2007-01-01", "hasta": "2018-12-31", "trades": total_trades, "pnl": "", "sharpe": sharpe_csv},
        hipotesis=h.id, etapa="1", sharpe_base="tasa_0",
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
    sharpe_car, total_trades, filas_tabla = ret[4], ret[5], ret[6]
    headers = ["pierna", "activo", "sharpe_fuera_muestra", "trades"] if len(h.piernas) > 1 else ["activo", "sharpe_fuera_muestra", "trades"]
    fila_cartera = ["cartera", "", sharpe_car, total_trades] if len(h.piernas) > 1 else ["cartera", sharpe_car, total_trades]
    click.echo(tabulate([*filas_tabla, fila_cartera], headers=headers, tablefmt="github"))
    if out:
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            if len(h.piernas) > 1:
                writer.writerow(["pierna", "activo", "sharpe_fuera_muestra", "trades", "params", "veredicto"])
                writer.writerows(
                    [*fila, json.dumps(parametros[(fila[0], fila[1])], sort_keys=True), veredicto]
                    for fila in filas_tabla
                )
            else:
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
        ret_filtro = filtrar_hipotesis(h, snapshot_name)
        resultados, parametros, veredicto, motivo = ret_filtro[:4]
        sharpe_car = ret_filtro[4] if len(ret_filtro) > 4 else None
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
            if len(h.piernas) > 1:
                motivo = "; ".join(dict.fromkeys(motivos)) if motivos else f"cartera pasa V30; control coincide en {len(filas)} de {len(filas)}"
                if veredicto == "PASA":
                    sharpe_motor = filas[0].get("sharpe_cartera") if filas else None
                    if alarma_cartera(sharpe_car, sharpe_motor):
                        veredicto = "ALARMA"
                        motivo = "filtro optimista en la cartera"
            else:
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
                    headers = ["pierna", "activo", "V30", "DSR"] if len(h.piernas) > 1 else ["activo", "V30", "DSR"]
                    filas_tab = [[f.get("estrategia", ""), f["activo"], f["v30"], f["dsr"]] if len(h.piernas) > 1 else [f["activo"], f["v30"], f["dsr"]] for f in filas]
                    click.echo(tabulate(filas_tab, headers=headers, tablefmt="github"))
                    motivos = [m for f in filas for m in f["motivos"]]
                    if len(h.piernas) > 1:
                        motivo = "; ".join(dict.fromkeys(motivos)) if motivos else "cartera pasa V30 y DSR"
                    else:
                        motivo = "; ".join(motivos)
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


def correr_cartera(archivo, snapshot_name, desde, hasta, carpeta):
    """Corre una cartera con los parámetros de su YAML y registra una fila en intentos.csv; devuelve su curva."""
    from quantagent.fabrica.validacion import _curva_y_v30_cartera, correr_motor

    h = cargar_hipotesis(archivo)
    params = {(p.clase, a): (p.combinaciones()[0] if p.parametros else {}) for p in h.piernas for a in p.activos}
    carpeta = Path(carpeta) / h.id

    filas = correr_motor(h, snapshot_name, params, desde, hasta, carpeta, "comparar")
    eq, _, motivos = _curva_y_v30_cartera(h, filas, carpeta, snapshot_name, "comparar", desde, hasta)
    if eq is None:
        raise ValueError(f"{h.id}: {'; '.join(motivos)}")
    return h.id, eq


@hipotesis_group.command("comparar", help="Comparar una cartera núcleo con la misma más una capa (ΔSharpe, ΔDSR).")
@click.argument("nucleo", type=click.Path(exists=True, dir_okay=False))
@click.argument("con_capa", type=click.Path(exists=True, dir_okay=False))
@click.option("--snapshot", "snapshot_name", required=True)
@click.option("--carpeta", type=click.Path(file_okay=False), required=True)
@click.option("--margen", type=float, required=True, help="ΔDSR mínimo para aceptar la capa; lo fija Fede antes de correr.")
@click.option("--desde", default="2007-01-01", show_default=True)
@click.option("--hasta", default="2018-12-31", show_default=True)
@click.option("--out", type=click.Path(dir_okay=False), help="Informe HTML de la comparación.")
def comparar_command(nucleo, con_capa, snapshot_name, carpeta, margen, desde, hasta, out):
    if hasta >= "2019-01-01":
        raise click.ClickException("La comparación no corre sobre la prueba (desde 2019-01-01)")
    try:
        id_n, eq_n = correr_cartera(nucleo, snapshot_name, desde, hasta, carpeta)
        try:
            id_con_capa = cargar_hipotesis(con_capa).id
        except Exception:
            id_con_capa = None
        dir_n = Path(carpeta) / id_n
        if id_con_capa and dir_n.exists():
            import shutil
            dir_c = Path(carpeta) / id_con_capa
            for sub in ["", "trades", "equity"]:
                src_sub = dir_n / sub if sub else dir_n
                dst_sub = dir_c / sub if sub else dir_c
                dst_sub.mkdir(parents=True, exist_ok=True)
                for f in src_sub.glob("*.csv"):
                    dst_file = dst_sub / f.name
                    if not dst_file.exists():
                        shutil.copy2(f, dst_file)
        id_c, eq_c = correr_cartera(con_capa, snapshot_name, desde, hasta, carpeta)

        c = comparar(eq_n, eq_c, snapshot_root() / "intentos.csv", margen)

    except (ValueError, SnapshotError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"{id_n} Sharpe {c['sharpe_nucleo']:.4f} DSR {c['dsr_nucleo']}; "
               f"{id_c} Sharpe {c['sharpe_capa']:.4f} DSR {c['dsr_capa']}")
    click.echo(f"ΔSharpe {c['delta_sharpe']:+.4f}; N {c['n_con_capa']} con la capa, {c['n_sin_capa']} sin ella")
    click.echo(f"Veredicto de la capa: {c['veredicto']} ({c['motivo']})")
    if out:
        Path(out).write_text(render_comparacion(c, id_n, id_c), encoding="utf-8")
    raise click.exceptions.Exit(int(c["veredicto"] != "ACEPTA"))
