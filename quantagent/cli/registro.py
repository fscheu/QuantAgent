"""Administración de ciclos de evaluación."""

import click

from quantagent.fabrica.registro import abrir_ciclo, cerrar_ciclo


@click.group("registro", help="Manage hypothesis evaluation cycles.",
             epilog="Examples:\n  quantagent registro ciclo abrir 2026-11 H1 H2\n  quantagent registro ciclo cerrar 2026-11")
def registro_group():
    pass


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
