"""Cartera de primas: pesos por igual riesgo (Carver) y retorno combinado."""
import json
from pathlib import Path

import pandas as pd


def pesos(cierres, piernas, ventana=60, seleccion=None):
    """Pesos fecha × (pierna, activo). Cada pierna pesa 1/S; dentro, 1/σ de los últimos `ventana` retornos.

    Se recalcula la primera sesión de cada mes con datos hasta la sesión previa.
    Sin `ventana` retornos disponibles (o σ no positivo), la pierna reparte por igual.

    `seleccion` (opcional): {pierna: panel 0/1 fecha × activo} para piernas transversales. En esas
    piernas el 1/S se reparte con 1/σ solo entre los activos elegidos ese día (la fila de la fecha de
    rebalanceo); los demás pesan 0, y sin elegidos la pierna queda en efectivo. Las piernas transversales
    cuentan en S. Las piernas sin entrada en `seleccion` no cambian.
    """
    seleccion = seleccion or {}
    columnas = pd.MultiIndex.from_tuples([(p, a) for p, activos in piernas.items() for a in activos])
    por_pierna = 1 / len(piernas)
    if cierres is None or isinstance(cierres, pd.Index):
        fila = {(p, a): por_pierna / len(activos) for p, activos in piernas.items() for a in activos}
        idx = cierres if isinstance(cierres, pd.Index) else [0]
        return pd.DataFrame([fila] * len(idx), index=idx, columns=columnas)
    sigma = cierres.pct_change().rolling(ventana).std().shift(1)
    inicio_mes = cierres.index.to_series().dt.to_period("M").ne(
        cierres.index.to_series().dt.to_period("M").shift())
    filas = {}
    for fecha in cierres.index[inicio_mes]:
        fila = {}
        for pierna, activos in piernas.items():
            if pierna in seleccion:
                elegidos = [a for a in activos if seleccion[pierna].loc[fecha, a] == 1]
            else:
                elegidos = list(activos)
            for activo in activos:
                fila[(pierna, activo)] = 0.0
            if not elegidos:
                continue
            s = sigma.loc[fecha, elegidos]
            inv = 1 / s if (s > 0).all() else pd.Series(1.0, index=elegidos)
            for activo in elegidos:
                fila[(pierna, activo)] = por_pierna * inv[activo] / inv.sum()
        filas[fecha] = fila
    return pd.DataFrame.from_dict(filas, orient="index", columns=columnas).reindex(cierres.index).ffill()


def pesos_objetivo(cierres, piernas, fecha, ventana=60, seleccion=None):
    """Pesos agregados por activo vigentes en `fecha`, con la lógica de `pesos`."""
    por_pierna = pesos(cierres, piernas, ventana, seleccion)
    fila = por_pierna.loc[:fecha].iloc[-1]
    return fila.T.groupby(level=1).sum().to_dict()


def guardar_sombra(archivo, cartera, fecha, pesos, valor):
    """Save a portfolio's target weights and value under its date."""
    archivo = Path(archivo)
    datos = json.loads(archivo.read_text()) if archivo.exists() else {}
    datos.setdefault(str(fecha), {})[cartera] = {
        "pesos": {symbol: float(weight) for symbol, weight in pesos.items()}, "valor": float(valor)
    }
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text(json.dumps(datos, indent=2, sort_keys=True) + "\n")


def leer_sombra(archivo, cartera, fecha):
    """Read the saved target weights and value for one portfolio date."""
    return json.loads(Path(archivo).read_text())[str(fecha)][cartera]


def combinar(retornos, pesos):
    """Retorno diario de la cartera: Σ peso × retorno de la pierna en cada activo."""
    return (retornos[pesos.columns] * pesos).sum(axis=1)


def curva_cartera(cierres, piernas, retornos_dict, capital=100_000.0, seleccion=None):
    """Calcula la curva de equity combinada con pesos por igual riesgo."""
    df_ret = pd.DataFrame(retornos_dict).fillna(0.0)
    if df_ret.empty:
        return None
    w = pesos(df_ret.index, piernas, seleccion=seleccion) if cierres is None else pesos(
        cierres, piernas, seleccion=seleccion).reindex(df_ret.index).ffill().bfill()
    ret = combinar(df_ret, w)
    return capital * (1 + ret).cumprod()


def evaluar_cartera(cierres, piernas, retornos_dict, capital=100_000.0, seleccion=None):
    """Combina retornos fuera de muestra con pesos por igual riesgo y devuelve sharpe a tasa 0."""
    from quantagent.backtesting.vbt_filter import equity_metrics

    eq = curva_cartera(cierres, piernas, retornos_dict, capital, seleccion)
    return None if eq is None else equity_metrics(eq, 0.0)["sharpe"]
