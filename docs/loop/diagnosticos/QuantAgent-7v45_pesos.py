"""Reproduce la discrepancia T4/curva de cartera observada el 2026-10-24."""

import pandas as pd

from quantagent.fabrica.cartera import combinar, curva_cartera, pesos, pesos_objetivo

dates = pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-04"])
closes = pd.DataFrame({"A": [100, 100, 110], "B": [100, 100, 100], "C": [100, 100, 100]}, index=dates)
legs = {0: ["A", "B", "C"]}
selection = pd.DataFrame({"A": [1, 1, 1], "B": [0, 0, 0], "C": [0, 0, 0]}, index=dates)
returns = {(0, "A"): pd.Series([0, 0, .1], index=dates),
           (0, "B"): pd.Series(0., index=dates), (0, "C"): pd.Series(0., index=dates)}
current = curva_cartera(closes, legs, returns).iloc[-1]
decided_weights = pesos(closes, legs, seleccion={0: selection})
decided = (100000 * (1 + combinar(pd.DataFrame(returns), decided_weights)).cumprod()).iloc[-1]
d1 = pesos_objetivo(closes, legs, dates[-1])
print(f"Current curve: {current:.2f}; selected-normalized T4: {decided:.2f}")
print(f"D1: {d1}; current exposure to selected A: {d1['A']:.6f}")
print(f"Real YAML equal-volatility example: current invested={.5 * 9 / 29 + .5 * 5 / 17:.6%}; T4=100%")
assert abs(current - 103333.33333333333) < .001
assert abs(decided - 110000) < .001
