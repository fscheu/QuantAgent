"""Medición de la cartera núcleo por régimen de mercado 2007–2018 (M01.2)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from tabulate import tabulate

from quantagent.data.snapshot import read_snapshot
from quantagent.fabrica.cartera import combinar, pesos
from quantagent.fabrica.filtro import filtrar_activo
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.regimen import clasificar_regimen, permutar_regimenes_bloques, sharpe_por_regimen
from quantagent.settings import COSTOS_PERFILES

HIPOTESIS = RAIZ / "hipotesis" / "cartera-tendencia-momentum.yaml"


def calcular_retornos_cartera(h, snapshot, desde="2007-01-01", hasta="2018-12-31"):
    cierres, ret = {}, {}
    for i, p in enumerate(h.piernas):
        for a in p.activos:
            v = read_snapshot(snapshot, a).loc[desde:hasta]
            cierres.setdefault(a, v["close"])
            ret[(i, a)] = filtrar_activo(p, v.rename_axis("timestamp").reset_index(), costos=COSTOS_PERFILES["etf"], capital=100_000.0)["retornos"]
    c_df, ret_df = pd.DataFrame(cierres), pd.DataFrame(ret).fillna(0.0)
    ret_c = combinar(ret_df, pesos(c_df, {0: h.piernas[0].activos, 1: h.piernas[1].activos}))
    ret_p = [combinar(pd.DataFrame({(i, a): ret[(i, a)] for a in p.activos}).fillna(0.0), pesos(c_df, {i: p.activos})) for i, p in enumerate(h.piernas)]
    return ret_c, ret_p


def contrastar_azar(ret, reg, n_rep=100, seed=42):
    def diff(r, g):
        sh = sharpe_por_regimen(r, g)
        return np.mean([v for k, v in sh.items() if k.startswith("alcista")] or [0.0]) - np.mean([v for k, v in sh.items() if k.startswith("bajista")] or [0.0])
    obs = diff(ret, reg)
    rng = np.random.default_rng(seed)
    sims = [diff(ret, permutar_regimenes_bloques(reg, seed=int(rng.integers(1e9)))) for _ in range(n_rep)]
    return obs, float(np.mean(sims)), float(np.mean(np.abs(sims) >= np.abs(obs)))


def medir(snapshot="etf-1d-2026-10-amplio", n_rep=100, seed=42):
    h = cargar_hipotesis(HIPOTESIS)
    ret_c, ret_p = calcular_retornos_cartera(h, snapshot)
    reg = clasificar_regimen(read_snapshot(snapshot, "SPY")["close"].loc["2007-01-01":"2018-12-31"])
    sh_c, sh_p = sharpe_por_regimen(ret_c, reg), [sharpe_por_regimen(rp, reg) for rp in ret_p]
    filas = [[k, sh_c.get(k, 0.0)] + [sp.get(k, 0.0) for sp in sh_p] for k in sorted(sh_c.keys())]
    print(tabulate(filas, headers=["Régimen", "Cartera"] + [p.clase for p in h.piernas], floatfmt=".2f"))

    obs_c, azar_c, p_c = contrastar_azar(ret_c, reg, n_rep=n_rep, seed=seed)
    diffs_p = [contrastar_azar(rp, reg, n_rep=n_rep, seed=seed) for rp in ret_p]
    print(f"\nContraste alcista vs bajista (N={n_rep}):\nCartera: obs={obs_c:+.2f}, azar={azar_c:+.2f}, p={p_c:.3f}")
    for i, p in enumerate(h.piernas):
        print(f"{p.clase}: obs={diffs_p[i][0]:+.2f}, azar={diffs_p[i][1]:+.2f}, p={diffs_p[i][2]:.3f}")

    mismo_signo = len({np.sign(dp[0]) for dp in diffs_p}) == 1
    decision = "se sigue" if (p_c < 0.05 and mismo_signo) else "se archiva"
    motivo = "diferencia supera el azar y piernas con mismo signo" if decision == "se sigue" else "diferencia no supera el azar o piernas con signos opuestos"
    print(f"\nConclusión: {decision} ({motivo}).")
    return {"tabla": filas, "p_val": p_c, "decision": decision}


if __name__ == "__main__":
    medir()
