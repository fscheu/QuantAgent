"""Comparador núcleo contra núcleo + capa (QuantAgent-d4cq.73, M05)."""
import csv
from pathlib import Path

from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.backtesting.verdict import dsr_contra_intentos
from quantagent.fabrica.criterio import anios_positivos, error_estandar_delta_sharpe


def _n_intentos(intentos_path):
    with Path(intentos_path).open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def comparar(eq_nucleo, eq_capa, intentos_path, margen):
    """ΔSharpe y ΔDSR de la capa. Ambos DSR usan el mismo `intentos.csv`, con las dos filas ya registradas.

    La capa se acepta solo si se cumplen las tres condiciones de QuantAgent-m9cx (Fede, 2026-10-20):
    ΔDSR supera `margen` (lo fija Fede antes de correr), ΔSharpe es positivo y la capa no pierde años
    con Sharpe positivo respecto del núcleo.
    """
    n = _n_intentos(intentos_path)
    sharpe_n, sharpe_c = (equity_metrics(eq, 0.0)["sharpe"] for eq in (eq_nucleo, eq_capa))
    dsr_n, dsr_c = (dsr_contra_intentos(eq, intentos_path) for eq in (eq_nucleo, eq_capa))
    ap_n, ap_c = anios_positivos(eq_nucleo), anios_positivos(eq_capa)
    delta_sharpe = sharpe_c - sharpe_n
    error_estandar = error_estandar_delta_sharpe(eq_capa, eq_nucleo)
    if dsr_n is None or dsr_c is None:
        delta_dsr, veredicto, motivo = None, "DESCARTA", "sin DSR en alguna de las dos carteras"
    else:
        delta_dsr = dsr_c - dsr_n
        fallas = []
        if delta_dsr <= margen:
            fallas.append(f"ΔDSR {delta_dsr:+.4f} <= margen {margen}")
        if delta_sharpe <= 0:
            fallas.append(f"ΔSharpe {delta_sharpe:+.4f} <= 0")
        if ap_c.positivos < ap_n.positivos:
            fallas.append(f"años positivos {ap_c.positivos} < {ap_n.positivos} del núcleo")
        veredicto = "DESCARTA" if fallas else "ACEPTA"
        motivo = "; ".join(fallas) if fallas else (
            f"ΔDSR {delta_dsr:+.4f} > margen {margen}; ΔSharpe {delta_sharpe:+.4f} > 0; "
            f"años positivos {ap_c.positivos} >= {ap_n.positivos}")
    return {"sharpe_nucleo": sharpe_n, "sharpe_capa": sharpe_c, "delta_sharpe": delta_sharpe,
            "dsr_nucleo": dsr_n, "dsr_capa": dsr_c, "delta_dsr": delta_dsr,
            "anios_positivos_nucleo": ap_n, "anios_positivos_capa": ap_c,
            "error_estandar_delta_sharpe": error_estandar,
            "n_con_capa": n, "n_sin_capa": n - 1, "margen": margen,
            "veredicto": veredicto, "motivo": motivo}
