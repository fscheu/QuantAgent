"""Piezas H4: conjunto H3, señales y gestión A, sin corrida pública ni registros."""

from pathlib import Path

import h3
import pandas as pd
import yaml

from quantagent.backtesting.intrabar_stops import check_intrabar_stops, check_stop_entry
from quantagent.backtesting.signals import bar_signals
from quantagent.strategy.arquetipo_a_strategy import ArquetipoAStrategy


def elegibilidad(w3, sesiones):
    """Activos habilitados por sesión según los pesos positivos de la semana vigente."""
    return w3.reindex(sesiones, method="ffill").fillna(0.0).gt(0).loc[h3.DESDE:h3.HASTA]


def senales_a(velas, simbolo):
    """Calienta A con historia previa y conserva todas las columnas de L24, aun fuera del conjunto."""
    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "hipotesis/arquetipo-a.yaml").read_text())
    estrategia = ArquetipoAStrategy(**{k: v[0] for k, v in spec["parametros"].items()})
    return bar_signals(estrategia, velas.loc[:h3.HASTA], simbolo, spec["timeframe"]).loc[h3.DESDE:]


def preparar_datos(cache="/tmp/ai-loop/h3-cache.pkl"):
    """Devuelve velas, pesos semanales, elegibilidad diaria y señales A para 2011–2018."""
    velas, w1, decisiones = h3.pesos_h1()
    etfs = {dec: [e for e in h3.scr.ETF_SECTOR if w1.loc[d, e] > 0] for d, dec in decisiones.items()}
    cand, _, _ = h3.candidatos(list(decisiones.values()), etfs, cache)
    w3 = h3.pesos_h3(w1, cand, decisiones)
    velas = {t: (velas[t] if t in velas else h3.read_snapshot(h3.ACC, t)).loc[:h3.HASTA] for t in w3.columns}
    habilitados = elegibilidad(w3, w1.index)
    senales = {t: senales_a(v, t) for t, v in velas.items()}
    return velas, w3, habilitados, senales


def orden_a(pendiente, vela, senal, habilitado):
    """Sin posición: devuelve (pendiente, entrada candidata); el consumidor admite y dimensiona."""
    if not habilitado:
        return None, None
    if pendiente is not None:
        precio = check_stop_entry(pendiente["entry_price"], vela)
        if precio is not None:
            return None, {"side": "long", "entry_price": precio,
                          "stop_loss": precio * (1 - pendiente["stop_loss_pct"]),
                          "maximo": precio, "salir": False}
        pendiente["restantes"] -= 1
        return (pendiente if pendiente["restantes"] > 0 else None), None
    if senal["entry"] == 1:
        pendiente = dict(senal, restantes=int(senal["entry_expiry_bars"]))
    return pendiente, None


def posicion_a(posicion, vela, senal):
    """Gestiona incluso fuera del conjunto; llamar también en la vela de entrada admitida."""
    if posicion["salir"]:
        return "TRAILING_STOP", float(vela["open"])
    salida = check_intrabar_stops(posicion, vela)
    if salida is not None:
        return salida
    # L24: primero el stop vigente intrabar; Chandelier se evalúa al cierre y llena al open siguiente.
    posicion["maximo"] = max(posicion["maximo"], float(vela["high"]))
    chandelier = posicion["maximo"] - senal["chandelier_dist"]
    posicion["salir"] = bool(vela["close"] <= chandelier)
    if not posicion["salir"]:
        posicion["stop_loss"] = max(posicion["stop_loss"], chandelier)
    return None


def curva(velas, habilitados, senales):
    """D2 desde 100.000: curva diaria neta, trades cerrados y posiciones aún abiertas.

    Entradas en orden de columnas; dimensionadas al open, sin liberar heat intradiario.
    Operado es volumen/equity al open; invertido incluye entradas y salidas en la misma vela.
    """
    costo = sum(h3.COSTOS_PERFILES["etf"].values())
    cierres = pd.DataFrame({t: v["close"] for t, v in velas.items()}).reindex(habilitados.index).ffill()
    aperturas = pd.DataFrame({t: v["open"] for t, v in velas.items()}).reindex(habilitados.index).fillna(cierres.shift())
    efectivo, posiciones, pendientes, trades, diario = 100_000.0, {}, {}, [], []
    for fecha, habilitado in habilitados.iterrows():
        equity = efectivo + sum(p["cantidad"] * aperturas.loc[fecha, t] for t, p in posiciones.items())
        base, volumen, costos, rechazos = equity, 0.0, 0.0, 0
        riesgo = sum(max(p["entry_price"] - p["stop_loss"], 0) * p["cantidad"] for p in posiciones.values())
        barras = {t: v.loc[fecha] for t, v in velas.items() if fecha in v.index and pd.notna(v.loc[fecha, "close"])}
        for t in habilitados.columns:
            if not habilitado[t]:
                pendientes.pop(t, None)
            if t in posiciones or t not in barras:
                continue
            pendientes[t], p = orden_a(pendientes.get(t), barras[t], senales[t].loc[fecha], habilitado[t])
            if p is None:
                continue
            distancia = p["entry_price"] - p["stop_loss"]
            cantidad = min(0.0025 * equity / distancia, 0.1 * equity / p["entry_price"])
            monto = cantidad * p["entry_price"]
            if riesgo + distancia * cantidad > 0.02 * (equity - monto * costo) + 1e-9:
                rechazos += 1
                continue
            posiciones[t] = dict(p, cantidad=cantidad, entrada=fecha)
            efectivo -= monto * (1 + costo)
            equity -= monto * costo
            riesgo += distancia * cantidad
            volumen, costos = volumen + monto, costos + monto * costo
        invertido = bool(posiciones)
        for t, p in list(posiciones.items()):
            salida = posicion_a(p, barras[t], senales[t].loc[fecha]) if t in barras else None
            if salida is None:
                continue
            motivo, precio = salida
            monto = p["cantidad"] * precio
            efectivo += monto * (1 - costo)
            volumen, costos = volumen + monto, costos + monto * costo
            trades.append(dict(p, simbolo=t, salida=fecha, exit_price=precio, motivo=motivo,
                               pnl=p["cantidad"] * (precio - p["entry_price"]) - costo * (monto + p["cantidad"] * p["entry_price"])))
            del posiciones[t]
        equity = efectivo + sum(p["cantidad"] * cierres.loc[fecha, t] for t, p in posiciones.items())
        heat = sum(max(p["entry_price"] - p["stop_loss"], 0) * p["cantidad"] for p in posiciones.values()) / equity
        diario.append(dict(equity=equity, invertido=invertido, heat=heat, rechazos=rechazos,
                           volumen=volumen, operado=volumen / base, costos=costos))
    return pd.DataFrame(diario, index=habilitados.index), trades, posiciones
