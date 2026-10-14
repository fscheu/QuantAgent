"""Veredictos sobre una tabla controlada, sin ejecutar backtests."""

import csv

import click
from tabulate import tabulate

from quantagent.backtesting.verdict import evaluar

METRICAS = ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")
LLENADOS = (("cierre", ""), ("apertura", "_apertura_siguiente"))
COLUMNAS = ["estrategia", "activo", "desde", "hasta", "veredicto", "motivos", "control"]
COLUMNAS += [campo + sufijo for _, sufijo in LLENADOS for campo in METRICAS]


@click.command("verdict")
@click.option("--tabla", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False, writable=True))
def verdict_command(tabla, out):
    with open(tabla, newline="") as f:
        reader = csv.DictReader(f)
        requeridas = [c for c in COLUMNAS if c not in ("veredicto", "motivos")]
        faltantes = [c for c in requeridas if c not in (reader.fieldnames or [])]
        if faltantes:
            raise click.ClickException("Faltan columnas: " + ", ".join(faltantes))
        filas = list(reader)
    for fila in filas:
        motivos = []
        for prefijo, sufijo in LLENADOS:
            valores = []
            for campo in METRICAS:
                texto = (fila[campo + sufijo] or "").strip()
                try:
                    valores.append((int(texto) if campo == "trades" else float(texto)) if texto else None)
                except ValueError:
                    raise click.ClickException(f"Valor inválido en {campo + sufijo}: {texto}")
            _, fallas = evaluar(*valores)
            motivos.extend(f"{prefijo}: {m}" for m in fallas)
        fila["veredicto"] = "NO PASA" if motivos else "PASA"
        fila["motivos"] = "; ".join(motivos)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)
    click.echo(tabulate([[f[c] for c in COLUMNAS] for f in filas], headers=COLUMNAS, tablefmt="github"))
    click.echo(f"PASA: {sum(f['veredicto'] == 'PASA' for f in filas)} de {len(filas)}")
