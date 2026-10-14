"""Veredictos sobre una tabla controlada, sin ejecutar backtests."""

import csv
import json
from pathlib import Path

import click
from tabulate import tabulate

from quantagent.backtesting.verdict import evaluar, montecarlo_bloques, p_valor_retornos

METRICAS = ("trades", "pnl", "sharpe_tasa_0", "max_drawdown")
LLENADOS = (("cierre", ""), ("apertura", "_apertura_siguiente"))
COLUMNAS = ["estrategia", "activo", "desde", "hasta", "veredicto", "motivos", "control"]
COLUMNAS += [campo + sufijo for _, sufijo in LLENADOS for campo in METRICAS]


@click.command("verdict")
@click.option("--tabla", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False, writable=True))
@click.option("--trades-dir", type=click.Path(exists=True, file_okay=False),
              help="Agrega concentración del PnL y PnL por año desde los CSV de trades.")
@click.option("--equity-dir", type=click.Path(exists=True, file_okay=False),
              help="Agrega p-valor, percentil final Monte Carlo y drawdown p95 informativos.")
def verdict_command(tabla, out, trades_dir, equity_dir):
    columnas = COLUMNAS + (["concentracion_pnl", "pnl_por_anio"] if trades_dir else [])
    columnas += ["p_valor", "mc_percentil_final", "mc_dd_p95"] if equity_dir else []
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
        if equity_dir:
            fila.update(p_valor="", mc_percentil_final="", mc_dd_p95="")
            archivo = Path(equity_dir) / f"{fila['estrategia']}-{fila['activo']}.csv"
            if archivo.exists():
                with archivo.open(newline="") as f:
                    equity = [float(r["equity"]) for r in csv.DictReader(f)]
                p_valor = p_valor_retornos(equity)
                if p_valor is not None:
                    fila["p_valor"] = f"{p_valor:.4f}"
                mc = montecarlo_bloques(equity)
                if mc is not None:
                    fila.update(mc_percentil_final=f"{mc[0]:.4f}", mc_dd_p95=f"{mc[1]:.4f}")
        if trades_dir:
            fila.update(concentracion_pnl="", pnl_por_anio="")
            archivo = Path(trades_dir) / f"{fila['estrategia']}-{fila['activo']}.csv"
            if archivo.exists():
                with archivo.open(newline="") as f:
                    trades = list(csv.DictReader(f))
                if trades:
                    pnls = [float(t["pnl"]) for t in trades]
                    total = sum(abs(p) for p in pnls)
                    fila["concentracion_pnl"] = round(max(abs(p) for p in pnls) / total, 4) if total else ""
                    anios = {}
                    for trade, pnl in zip(trades, pnls):
                        anio = trade["exit_time"][:4]
                        anios[anio] = anios.get(anio, 0) + pnl
                    fila["pnl_por_anio"] = "{" + ", ".join(
                        f"{json.dumps(anio)}: {anios[anio]:.2f}" for anio in sorted(anios)
                    ) + "}"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)
    click.echo(tabulate([[f[c] for c in columnas] for f in filas], headers=columnas, tablefmt="github"))
    click.echo(f"PASA: {sum(f['veredicto'] == 'PASA' for f in filas)} de {len(filas)}")
