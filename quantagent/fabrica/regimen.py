"""Clasificación de régimen de mercado y análisis de rendimiento por régimen."""
import numpy as np
import pandas as pd


def clasificar_regimen(
    cierres_spy: pd.Series,
    ventana_vol: int = 20,
    mediana_vol_dias: int = 252,
    ventana_media: int = 200,
) -> pd.Series:
    """Clasifica régimen de SPY con volatilidad 20d y SMA 200d con datos hasta t-1 (shift(1))."""
    ret = cierres_spy.pct_change()
    vol = ret.rolling(ventana_vol).std()
    med_vol = vol.rolling(mediana_vol_dias).median()
    sma = cierres_spy.rolling(ventana_media).mean()

    valido = vol.notna() & med_vol.notna() & sma.notna()
    alta_vol = vol > med_vol
    alcista = cierres_spy >= sma

    reg = pd.Series(index=cierres_spy.index, dtype=object)
    reg[valido & alcista & ~alta_vol] = "alcista_baja_vol"
    reg[valido & alcista & alta_vol] = "alcista_alta_vol"
    reg[valido & ~alcista & ~alta_vol] = "bajista_baja_vol"
    reg[valido & ~alcista & alta_vol] = "bajista_alta_vol"
    return reg.shift(1)


def sharpe_por_regimen(
    retornos: pd.Series,
    regimenes: pd.Series,
    risk_free_rate: float = 0.0,
    anualizar: bool = True,
) -> dict[str, float]:
    """Calcula el ratio de Sharpe diario por cada categoría de régimen."""
    df = pd.DataFrame({"ret": retornos, "reg": regimenes}).dropna()
    ppy = 252.0 if anualizar else 1.0
    factor = np.sqrt(ppy)
    resultado = {}
    for cat, grupo in df.groupby("reg"):
        ret = grupo["ret"]
        if len(ret) < 2 or ret.std() == 0:
            resultado[str(cat)] = 0.0
        else:
            sr = (ret.mean() - risk_free_rate / ppy) / ret.std() * factor
            resultado[str(cat)] = float(sr)
    return resultado


def permutar_regimenes_bloques(regimenes: pd.Series, seed: int | None = None) -> pd.Series:
    """Permuta bloques mensuales de etiquetas para contraste contra ruido conservando frecuencias."""
    if regimenes.empty:
        return regimenes.copy()
    idx = pd.to_datetime(regimenes.index)
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    grupos = [g.to_numpy() for _, g in regimenes.groupby(idx_naive.to_period("M"), sort=False)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(grupos))
    valores = np.concatenate([grupos[i] for i in perm])
    return pd.Series(valores, index=regimenes.index, name=regimenes.name)


def price_momentum_score(
    cierres: pd.DataFrame,
    ventanas: tuple[int, ...] = (21, 63, 126),
    ventana_vol: int = 60,
) -> pd.DataFrame:
    """PriceMomentumScore: promedio del retorno en ventanas k sobre volatilidad de ventana_vol sesiones."""
    if not ventanas:
        return pd.DataFrame(np.nan, index=cierres.index, columns=cierres.columns)
    ret_diario = cierres.pct_change()
    vol = ret_diario.rolling(ventana_vol).std(ddof=1)
    scores = [(cierres / cierres.shift(k) - 1.0) / vol for k in ventanas]
    res = scores[0]
    for s in scores[1:]:
        res = res + s
    return res / len(ventanas)


def filtro_caida(
    score: pd.DataFrame,
    cierres_spy: pd.Series,
    ventana_media: int = 200,
    pendiente: int = 20,
) -> pd.DataFrame:
    """Filtro de caída: True si score < 0 y SPY < SMA(ventana_media) con SMA decreciente en ventana de pendiente."""
    sma = cierres_spy.rolling(ventana_media).mean()
    sma_prev = sma.shift(pendiente)
    caida_spy = (cierres_spy < sma) & (sma < sma_prev)
    caida_spy = caida_spy.reindex(score.index, fill_value=False).fillna(False).astype(bool)
    return (score < 0).fillna(False).mul(caida_spy, axis=0).astype(bool)



def z_temas(z_series: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """z de cada tema: promedio de los z de sus series, con el signo invertido donde el YAML lo pide."""
    temas = {}
    for tema, d in priors["temas"].items():
        cols = [-z_series[s["serie"]] if s.get("invertir") else z_series[s["serie"]] for s in d["series"]]
        temas[tema] = pd.concat(cols, axis=1).mean(axis=1, skipna=False)
    return pd.DataFrame(temas, index=z_series.index)


def macro_theme_score(z_tema: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """MacroThemeScore por activo: Σ signo × z_tema / cantidad de temas con signo ≠ 0 y z disponible (0 si no hay)."""
    signos = pd.DataFrame(0.0, index=list(priors["temas"]), columns=priors["universo"])
    for tema, d in priors["temas"].items():
        for direccion, valor in (("positivo", 1.0), ("negativo", -1.0)):
            if direccion in d["signos"]:
                signos.loc[tema, d["signos"][direccion]["activos"]] = valor
    z = z_tema[signos.index]
    suma = z.fillna(0.0) @ signos
    cuenta = z.notna().astype(float) @ (signos != 0).astype(float)
    return (suma / cuenta.where(cuenta > 0)).fillna(0.0)


def z_series_macro(priors: dict, sesiones, base_dir=None) -> pd.DataFrame:
    """z móvil de la transformada punto en el tiempo de cada serie macro de los priors, por sesión."""
    from quantagent.data.macro import transformada_punto_en_el_tiempo, zscore_movil

    series = {s["serie"]: s["transformada"] for d in priors["temas"].values() for s in d["series"]}
    return pd.DataFrame({
        serie: zscore_movil(
            transformada_punto_en_el_tiempo(priors["snapshot_macro"], serie, sesiones, tipo, base_dir),
            priors["ventana_z_meses"],
        )
        for serie, tipo in series.items()
    })


def _z_transversal(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1, ddof=1), axis=0)


def sector_score(macro: pd.DataFrame, momentum: pd.DataFrame, peso_macro: float) -> pd.DataFrame:
    """SectorScore = peso_macro × z(macro) + (1 - peso_macro) × z(momentum), z transversal por fecha. NaN donde momentum es NaN."""
    z_macro = _z_transversal(macro.where(momentum.notna())).fillna(0.0)
    z_mom = _z_transversal(momentum).fillna(0.0)
    return (peso_macro * z_macro + (1.0 - peso_macro) * z_mom).where(momentum.notna())


def regimen_semanal(cierres: pd.DataFrame, priors: dict, peso_macro: float, base_dir=None) -> pd.DataFrame:
    """Una fila por última sesión de cada semana desde 2006: z por tema, score por activo, habilitados, bloqueados y estado."""
    cierres = cierres[priors["universo"]]
    momentum = price_momentum_score(cierres)
    ultimas = cierres.index.to_series().groupby(cierres.index.to_period("W")).max()
    semanas = pd.DatetimeIndex(ultimas[ultimas >= "2006-01-01"])
    z_tema = z_temas(z_series_macro(priors, semanas, base_dir), priors)
    mom = momentum.loc[semanas]
    score = sector_score(macro_theme_score(z_tema, priors), mom, peso_macro)
    bloqueo = filtro_caida(mom, cierres["SPY"])
    risk_off = filtro_caida(pd.DataFrame({"SPY": -1.0}, index=semanas), cierres["SPY"])["SPY"]
    filas = []
    for t in semanas:
        elegibles = score.loc[t][(score.loc[t] > 0) & ~bloqueo.loc[t]].sort_values(ascending=False, kind="stable")
        estado = "risk-off" if risk_off[t] else "neutral" if z_tema.loc[t, "estres"] > 1 else "risk-on"
        filas.append({
            "fecha": t,
            "version": priors["version"],
            **{f"z_{c}": z_tema.loc[t, c] for c in z_tema},
            **{f"score_{a}": score.loc[t, a] for a in score},
            "habilitados": ",".join(elegibles.index[:3]),
            "bloqueados": ",".join(bloqueo.columns[bloqueo.loc[t]]),
            "estado": estado,
        })
    return pd.DataFrame(filas)
