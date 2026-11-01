"""Simulación de carteras de hipótesis con acciones enteras."""

import math
import statistics

import pandas as pd

from quantagent.fabrica.transversal import semanales
from quantagent.trading.cartera_paper import pesos_operables
from quantagent.trading.rebalanceador import ordenes_rebalanceo


def simular_cartera_cuenta_chica(hipotesis, cierres, capital, slippage_pct=0.0,
                                  commission_pct=0.0, costo_entrada_usd=0.0,
                                  dividend_rates=None, withholding=0.0, macro=None):
    """Simula mensual/semanalmente una hipótesis sobre cierres ajustados."""
    dividend_rates = dividend_rates or {}
    posiciones, cash = {}, float(capital) - costo_entrada_usd
    pagos = float(costo_entrada_usd)
    curve, zeros = [], set()
    calendario = cierres.index.to_series().dt.to_period("M").ne(cierres.index.to_series().dt.to_period("M").shift())
    semanas = cierres.index.to_series().apply(lambda d: d.isocalendar()[:2])
    rebalanceos = calendario | semanas.ne(semanas.shift()) if semanales(hipotesis.piernas) else calendario
    for i, (fecha, row) in enumerate(cierres.iterrows()):
        precios = row.to_dict()
        posiciones_previas = posiciones.copy()
        precios_previos = cierres.iloc[i - 1].to_dict() if i else precios
        if rebalanceos.loc[fecha]:
            pesos = pesos_operables(hipotesis, cierres.loc[:fecha], fecha, macro=macro)
            if pesos.attrs.get("bloqueos"):
                raise ValueError("No se pudieron calcular pesos operables: " + "; ".join(pesos.attrs["bloqueos"]))
            equity = cash + sum(qty * precios[s] for s, qty in posiciones.items())
            ordenes = ordenes_rebalanceo(pesos.to_dict(), {s: {"qty": q} for s, q in posiciones.items()}, precios, equity * .95)
            for symbol, side, qty, price in ordenes:
                notional = qty * price
                slip = slippage_pct.get(symbol, 0.0) if isinstance(slippage_pct, dict) else slippage_pct
                rate = slip + commission_pct
                fee = notional * rate
                if side.value == "sell":
                    cash += notional - fee
                    posiciones[symbol] -= qty
                else:
                    cash -= notional + fee
                    posiciones[symbol] = posiciones.get(symbol, 0) + qty
                pagos += fee
            for symbol, weight in pesos.items():
                if weight > 0 and math.floor(weight * equity * .95 / precios[symbol]) == 0:
                    zeros.add(symbol)
        taxes = sum(qty * precios_previos[s] * dividend_rates.get(s, {}).get(fecha.date().isoformat(), 0) * withholding
                    for s, qty in posiciones_previas.items())
        cash -= taxes
        pagos += taxes
        equity = cash + sum(qty * precios[s] for s, qty in posiciones.items())
        curve.append({"date": fecha, "equity": equity, "positions_value": sum(q * precios[s] for s, q in posiciones.items())})
    returns = [b["equity"] / a["equity"] - 1 for a, b in zip(curve, curve[1:]) if a["equity"] > 0]
    years = (cierres.index[-1] - cierres.index[0]).days / 365.25
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0
    sharpe = statistics.mean(returns) / deviation * math.sqrt(len(returns) / years) if deviation and years > 0 else 0
    return {"equity_curve": curve, "return": curve[-1]["equity"] / capital - 1,
            "sharpe": sharpe, "costs": pagos, "zero_positions": sorted(zeros),
            "concentration": max((p["positions_value"] / p["equity"] for p in curve if p["equity"]), default=0)}
