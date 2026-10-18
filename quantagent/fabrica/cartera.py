"""Cartera de primas: pesos por igual riesgo (Carver) y retorno combinado."""
import pandas as pd


def pesos(cierres, piernas, ventana=60):
    """Pesos fecha × (pierna, activo). Cada pierna pesa 1/S; dentro, 1/σ de los últimos `ventana` retornos.

    Se recalcula la primera sesión de cada mes con datos hasta la sesión previa.
    Sin `ventana` retornos disponibles (o σ no positivo), la pierna reparte por igual.
    """
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
            s = sigma.loc[fecha, activos]
            inv = 1 / s if (s > 0).all() else pd.Series(1.0, index=activos)
            for activo in activos:
                fila[(pierna, activo)] = por_pierna * inv[activo] / inv.sum()
        filas[fecha] = fila
    return pd.DataFrame.from_dict(filas, orient="index", columns=columnas).reindex(cierres.index).ffill()


def combinar(retornos, pesos):
    """Retorno diario de la cartera: Σ peso × retorno de la pierna en cada activo."""
    return (retornos[pesos.columns] * pesos).sum(axis=1)


def curva_cartera(cierres, piernas, retornos_dict, capital=100_000.0):
    """Calcula la curva de equity combinada con pesos por igual riesgo."""
    df_ret = pd.DataFrame(retornos_dict).fillna(0.0)
    if df_ret.empty:
        return None
    w = pesos(df_ret.index, piernas) if cierres is None else pesos(cierres, piernas).reindex(df_ret.index).ffill().bfill()
    ret = combinar(df_ret, w)
    return capital * (1 + ret).cumprod()


def evaluar_cartera(cierres, piernas, retornos_dict, capital=100_000.0):
    """Combina retornos fuera de muestra con pesos por igual riesgo y devuelve sharpe a tasa 0."""
    from quantagent.backtesting.vbt_filter import equity_metrics

    eq = curva_cartera(cierres, piernas, retornos_dict, capital)
    return None if eq is None else equity_metrics(eq, 0.0)["sharpe"]
