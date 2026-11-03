"""Administración de ciclos de evaluación."""

import click
from datetime import date

from quantagent.default_config import CICLOS_PARA_REVISAR
from quantagent.fabrica.registro import (
    abrir_ciclo,
    calendario_ciclos,
    cerrar_ciclo,
    ciclos_sin_etapa3,
    importar_csv,
)
from quantagent.fabrica.registro_vista import informe_ciclo


@click.group("registro", help="Manage hypothesis evaluation cycles.",
             epilog="Examples:\n  quantagent registro ciclo abrir 2026-11 H1 H2\n  quantagent registro ciclo cerrar 2026-11")
def registro_group():
    pass


@registro_group.command("corte", help="Show consecutive closed cycles without a stage 3 pass.",
                        epilog="Example: quantagent registro corte")
def corte_command():
    cuenta = ciclos_sin_etapa3()
    click.echo(f"Ciclos cerrados consecutivos sin pasar etapa 3: {cuenta}")
    if cuenta >= CICLOS_PARA_REVISAR:
        click.echo(f"Se llegó a {CICLOS_PARA_REVISAR}: revisar el enfoque.")
    else:
        click.echo(f"Todavía no se llegó a {CICLOS_PARA_REVISAR} ciclos; no hace falta revisar.")


@registro_group.command("importar", help="Import hypothesis stages and attempts from snapshot CSV files.",
                        epilog="Example: quantagent registro importar --dry-run")
@click.option("--dry-run", is_flag=True, help="Preview counts without writing to the hypothesis registry.")
def importar_command(dry_run):
    resultado = importar_csv(dry_run=dry_run)
    etapas, intentos = resultado["etapas"], resultado["intentos"]
    if dry_run:
        click.echo(f"(dry-run) agregaría etapas {etapas['agregados']} de {etapas['leidos']}; "
                   f"intentos {intentos['agregados']} de {intentos['leidos']}; omitidas {resultado['omitidas']}")
    else:
        click.echo(f"Importación: etapas {etapas['agregados']} agregadas de {etapas['leidos']}; "
                   f"intentos {intentos['agregados']} agregados de {intentos['leidos']}; omitidas {resultado['omitidas']}")


@registro_group.command("calendario", help="Show monthly hypothesis cycle dates and registry status.",
                        epilog="Example: quantagent registro calendario 2026-11")
@click.argument("desde", required=False, default=None)
def calendario_command(desde):
    desde = desde or date.today().strftime("%Y-%m")
    try:
        ciclos = calendario_ciclos(desde)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    for ciclo in ciclos:
        linea = (f"{ciclo['ciclo']}: propuesta {ciclo['propuesta']}; "
                 f"embudo {ciclo['embudo']}; revisión paper {ciclo['revision_paper']}; "
                 f"{ciclo['estado']}")
        if ciclo["estado"] == "cerrado":
            linea += f" (promovidas: {ciclo['promovidas']})"
        click.echo(linea)
    cuenta = ciclos_sin_etapa3()
    click.echo(f"Corte: {cuenta}/{CICLOS_PARA_REVISAR} ciclos cerrados consecutivos sin pasar etapa 3.")
    if cuenta >= CICLOS_PARA_REVISAR:
        click.echo("Toca revisar el enfoque.")
    else:
        click.echo("Todavía no toca revisar el enfoque.")


@registro_group.command("ver", help="List hypothesis registry events in chronological order.",
                        epilog="Examples:\n  quantagent registro ver\n  quantagent registro ver --hipotesis H1")
@click.option("--hipotesis", "hipotesis_id", help="Show events for this hypothesis ID.")
def ver_command(hipotesis_id):
    from quantagent.fabrica.registro import _leer_eventos, _fecha

    eventos, _ = _leer_eventos()
    if hipotesis_id is not None:
        eventos = [e for e in eventos if e.get("hipotesis") == hipotesis_id]
    eventos = sorted(enumerate(eventos), key=lambda item: (_fecha(item[1]), item[0]))
    if not eventos:
        click.echo("No hay eventos en el registro.")
        return
    for _, evento in eventos:
        click.echo(" | ".join(str(evento.get(campo) or "") for campo in
                              ("fecha_utc", "hipotesis", "evento", "etapa", "veredicto", "ciclo", "motivo")))


@registro_group.group("ciclo", help="Open or close a monthly hypothesis cycle.",
                      epilog="Example: quantagent registro ciclo abrir 2026-11 H1 H2")
def ciclo_group():
    pass


@ciclo_group.command("abrir", help="Open a cycle with up to five hypothesis IDs.",
                     epilog="Example: quantagent registro ciclo abrir 2026-11 H1 H2")
@click.argument("ciclo")
@click.argument("hipotesis", nargs=-1, required=True)
def abrir_command(ciclo, hipotesis):
    try:
        abrir_ciclo(ciclo, hipotesis)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Ciclo abierto: {ciclo} ({len(hipotesis)} hipótesis)")


@ciclo_group.command("cerrar", help="Close an open cycle and save its immutable receipt.",
                     epilog="Example: quantagent registro ciclo cerrar 2026-11")
@click.argument("ciclo")
def cerrar_command(ciclo):
    try:
        recibo = cerrar_ciclo(ciclo)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Ciclo cerrado: {ciclo}; promovidas: {recibo['promovidas']}")


@ciclo_group.command("informe", help="Show a phone-friendly report for a monthly cycle.",
                     epilog="Example: quantagent registro ciclo informe 2026-11")
@click.argument("ciclo")
def informe_command(ciclo):
    try:
        texto = informe_ciclo(ciclo)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(texto)
