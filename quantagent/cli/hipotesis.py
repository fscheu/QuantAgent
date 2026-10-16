"""CLI de admisión de hipótesis (QuantAgent-d4cq.14.1)."""

import click
from tabulate import tabulate

from quantagent.data.snapshot import SnapshotError, read_snapshot
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.filtro import filtrar_activo
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.settings import COSTOS_PERFILES

from .sweep import registrar_intento

UMBRAL_SHARPE_ETAPA1 = 0.10
MAX_ACTIVOS_BAJO_UMBRAL = 8



@click.group("hipotesis", help="Evaluar hipótesis de la fábrica de estrategias.")
def hipotesis_group():
    pass


@hipotesis_group.command("filtrar", help="Cargar y admitir una hipótesis sobre 2007–2018.")
@click.argument("archivo", type=click.Path(exists=True, dir_okay=False))
@click.option("--snapshot", "snapshot_name", required=True)
def filtrar_command(archivo, snapshot_name):
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
        for motivo in motivos:
            click.echo(motivo)
        raise click.exceptions.Exit(1)
    resultados = []
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
        registrar_intento(
            h.clase, activo, snapshot_name, "etf", 1.0, h.parametros,
            {"desde": "2007-01-01", "hasta": "2018-12-31", "trades": r["trades"],
             "pnl": "", "sharpe": r["sharpe_tasa_0"]}, hipotesis=h.id, etapa="1",
        )
    click.echo(tabulate(resultados, headers=["activo", "sharpe_fuera_muestra", "trades"], tablefmt="github"))
    bajos = sum(r[1] < UMBRAL_SHARPE_ETAPA1 for r in resultados)
    click.echo("CORTA" if bajos > MAX_ACTIVOS_BAJO_UMBRAL else "PASA")
