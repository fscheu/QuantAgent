"""Administración de ciclos de evaluación."""

import click

from quantagent.default_config import CICLOS_PARA_REVISAR
from quantagent.fabrica.registro import abrir_ciclo, cerrar_ciclo, ciclos_sin_etapa3


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
