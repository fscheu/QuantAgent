"""Comparador núcleo contra núcleo + capa (QuantAgent-d4cq.73, M05)."""
import csv
from pathlib import Path

from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.backtesting.verdict import dsr_contra_intentos


def _n_intentos(intentos_path):
    with Path(intentos_path).open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def comparar(eq_nucleo, eq_capa, intentos_path, margen):
    """ΔSharpe y ΔDSR de la capa. Ambos DSR usan el mismo `intentos.csv`, con las dos filas ya registradas.

    La capa se acepta solo si ΔDSR supera `margen`, que fija Fede antes de correr.
    """
    n = _n_intentos(intentos_path)
    sharpe_n, sharpe_c = (equity_metrics(eq, 0.0)["sharpe"] for eq in (eq_nucleo, eq_capa))
    dsr_n, dsr_c = (dsr_contra_intentos(eq, intentos_path) for eq in (eq_nucleo, eq_capa))
    if dsr_n is None or dsr_c is None:
        delta_dsr, veredicto, motivo = None, "DESCARTA", "sin DSR en alguna de las dos carteras"
    else:
        delta_dsr = dsr_c - dsr_n
        acepta = delta_dsr > margen
        veredicto = "ACEPTA" if acepta else "DESCARTA"
        motivo = f"ΔDSR {delta_dsr:+.4f} {'>' if acepta else '<='} margen {margen}"
    return {"sharpe_nucleo": sharpe_n, "sharpe_capa": sharpe_c, "delta_sharpe": sharpe_c - sharpe_n,
            "dsr_nucleo": dsr_n, "dsr_capa": dsr_c, "delta_dsr": delta_dsr,
            "n_con_capa": n, "n_sin_capa": n - 1, "margen": margen,
            "veredicto": veredicto, "motivo": motivo}
