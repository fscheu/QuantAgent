"""Cartera de primas: pesos por igual riesgo (Carver) y retorno combinado."""
import json
from pathlib import Path

import pandas as pd


def pesos_f1(cierres):
    """Pesos SPY/IEF por fecha para F1; el desvío 60/40 usa cierres hasta la sesión previa."""
    resultado = pd.DataFrame(0.5, index=cierres.index, columns=["SPY", "IEF"])
    meses = cierres.index.to_period("M")
    finales = set()
    for mes in meses.unique():
        fechas = cierres.index[meses == mes]
        finales.update(fechas[-5:])
    for i, fecha in enumerate(cierres.index):
        if fecha not in finales or i == 0:
            continue
        mes_inicio = meses[i]
        anteriores = [j for j in range(i) if meses[j] < mes_inicio]
        if not anteriores:
            continue
        ancla = cierres.iloc[anteriores[-1]]
        previo = cierres.iloc[i - 1]
        spy = 0.6 * previo["SPY"] / ancla["SPY"]
        ief = 0.4 * previo["IEF"] / ancla["IEF"]
        peso_spy = spy / (spy + ief)
        resultado.loc[fecha] = (1.0, 0.0) if peso_spy < 0.6 else (0.0, 1.0) if peso_spy > 0.6 else (0.5, 0.5)
    return resultado


def pesos(cierres, piernas, ventana=60, seleccion=None, semanales=()):
    """Pesos fecha × (pierna, activo). Cada pierna pesa 1/S; dentro, 1/σ de los últimos `ventana` retornos.

    Se recalcula la primera sesión de cada mes con datos hasta la sesión previa; las piernas en `semanales`
    (índices de pierna con regla de decisión semanal) además la primera sesión de cada semana.
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
    semana = cierres.index.to_series().apply(lambda d: d.isocalendar()[:2])
    inicio_semana = semana.ne(semana.shift())
    filas, fila = {}, {}
    for fecha in cierres.index[inicio_mes | inicio_semana if semanales else inicio_mes]:
        fila = dict(fila)
        for pierna, activos in piernas.items():
            if not (inicio_mes[fecha] or pierna in semanales):
                continue
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


def pesos_h3(pesos_etf, candidatos, decisiones):
    """Pesos de H3 (M5-33): cada ETF sectorial de `pesos_etf` (fecha × ETF, pesos de H1) pasa a sus acciones.

    `decisiones`: {fecha de rebalanceo: fecha de decisión de `candidatos_semanales`}. El peso del ETF se reparte
    por igual entre sus candidatos de rank 1-3 en esa fecha (los que haya); sin candidatos, o si el ETF no es
    sectorial (EEM, EWZ, SPY: sin rank), queda el ETF. Devuelve fecha de rebalanceo × activo (ETFs y acciones).
    """
    filas = {}
    for fecha, decision in decisiones.items():
        fila = {}
        for etf, peso in pesos_etf.loc[fecha][lambda s: s > 0].items():
            elegidos = candidatos[(candidatos["fecha"] == decision) & (candidatos["etf"] == etf) & (candidatos["rank"] <= 3)]
            for activo in elegidos["simbolo"].tolist() or [etf]:
                fila[activo] = peso / max(len(elegidos), 1)
        filas[fecha] = fila
    return pd.DataFrame.from_dict(filas, orient="index").reindex(list(decisiones)).fillna(0.0)  # semana sin activos = efectivo


def combinar(retornos, pesos):
    """Retorno diario de la cartera: Σ peso × retorno de la pierna en cada activo."""
    return (retornos[pesos.columns] * pesos).sum(axis=1)


def curva_cartera(cierres, piernas, retornos_dict, capital=100_000.0, seleccion=None, semanales=()):
    """Calcula la curva de equity combinada con pesos por igual riesgo."""
    df_ret = pd.DataFrame(retornos_dict).fillna(0.0)
    if df_ret.empty:
        return None
    w = pesos(df_ret.index, piernas, seleccion=seleccion) if cierres is None else pesos(
        cierres, piernas, seleccion=seleccion, semanales=semanales).reindex(df_ret.index).ffill().bfill()
    ret = combinar(df_ret, w)
    return capital * (1 + ret).cumprod()


def evaluar_cartera(cierres, piernas, retornos_dict, capital=100_000.0, seleccion=None, semanales=()):
    """Combina retornos fuera de muestra con pesos por igual riesgo y devuelve sharpe a tasa 0."""
    from quantagent.backtesting.vbt_filter import equity_metrics

    eq = curva_cartera(cierres, piernas, retornos_dict, capital, seleccion, semanales)
    return None if eq is None else equity_metrics(eq, 0.0)["sharpe"]
