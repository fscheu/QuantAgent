"""Piernas transversales: ranking de activos del universo, un tercio por arriba (QuantAgent P05.1)."""
from pathlib import Path

import pandas as pd
import yaml

from quantagent.fabrica.regimen import macro_theme_score, price_momentum_score, sector_score, z_series_macro, z_temas
from quantagent.strategy.momentum_12m_strategy import is_last_session_of_month

# Registro propio de métricas transversales: nombre de clase del YAML -> función pura del panel de cierres
# (fecha × activo) que devuelve un panel de la misma forma donde MAYOR es mejor (la baja volatilidad devuelve
# -σ). La métrica solo puede mirar filas hasta la fecha de cada fila; NaN = historia insuficiente.
# Lo llenan P06 y P07; las clases transversales no son `TradingStrategy`, por eso no van en `registry.py`.
METRICAS = {}

# Universo de la pierna de baja volatilidad (P07): los ETFs de acciones del universo amplio. Se define por clase
# de activo, no por resultados: rankear bonos contra acciones mide otra cosa ("baja volatilidad" sería "bonos").
# Mismo criterio que `*acciones` de catalogo/primas.yaml; el test lo verifica. Excluye bonos, materias primas,
# divisas y VNQ (inmobiliario).
UNIVERSO_ACCIONES = frozenset(
    {"EEM", "EFA", "EWJ", "EWZ", "FXI", "IWM", "QQQ", "SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"}
)
# Con menos de 6 ETFs el tercio es 1 solo y la pierna queda demasiado concentrada (decisión piernas-transversales T5).
MIN_ACTIVOS_PIERNA = 6
VENTANA_VOLATILIDAD = 252  # sesiones; constante fija, 0 parámetros libres


def es_transversal(clase):
    """True si `clase` es una métrica transversal registrada (no una `TradingStrategy` de `registry.py`)."""
    return clase in METRICAS


def leer_seleccion(carpeta, piernas, indice=None):
    """Lee paneles guardados por pierna; las fechas sin señal quedan en cero."""
    seleccion = {}
    for i, pierna in enumerate(piernas):
        if es_transversal(pierna.clase):
            columnas = {}
            for activo in pierna.activos:
                archivo = Path(carpeta) / f"senal-{pierna.clase}-{activo}.csv"
                columnas[activo] = pd.read_csv(archivo, parse_dates=["timestamp"]).set_index("timestamp")["posicion"]
            seleccion[i] = pd.DataFrame(columnas).reindex(index=indice).fillna(0)
    return seleccion


def baja_volatilidad(cierres):
    """-σ de los retornos diarios de las últimas 252 sesiones (Ang et al. 2006): el más calmo puntúa más alto.

    Pide la ventana completa (NaN si falta algún retorno). Solo tiene sentido dentro de una clase de activo:
    rankeando bonos y acciones juntos, "baja volatilidad" sería casi siempre "bonos"; el universo es `UNIVERSO_ACCIONES`.
    """
    retornos = cierres.pct_change(fill_method=None)
    return -retornos.rolling(VENTANA_VOLATILIDAD, min_periods=VENTANA_VOLATILIDAD).std()


METRICAS["baja-volatilidad"] = baja_volatilidad


SALTO_MOMENTUM = 21  # un mes de sesiones: el "menos uno" de la literatura
VENTANA_MOMENTUM = 252  # doce meses de sesiones


def momentum_12m_menos_1m(cierres):
    """Retorno de 12 meses salteando el último mes: cierre[t-21] / cierre[t-252] - 1 (QuantAgent P06).

    Jegadeesh y Titman (1993) y Asness, Moskowitz y Pedersen (2013). Saltear el último mes evita el rebote de
    corto plazo. Usa solo filas anteriores o iguales a la de cada fecha; NaN si falta alguno de los dos cierres.
    """
    return cierres.shift(SALTO_MOMENTUM) / cierres.shift(VENTANA_MOMENTUM) - 1


METRICAS["momentum-transversal"] = momentum_12m_menos_1m

PRIORS_M5 = yaml.safe_load((Path(__file__).parents[2] / "catalogo" / "m5-priors.yaml").read_text(encoding="utf-8"))


def _sector_score(peso_macro):
    def metrica(cierres):
        cierres = cierres[[a for a in PRIORS_M5["universo"] if a in cierres.columns]]
        if peso_macro == 0:
            macro = pd.DataFrame(0.0, index=cierres.index, columns=cierres.columns)
        else:
            z_tema = z_temas(z_series_macro(PRIORS_M5, cierres.index), PRIORS_M5)
            macro = macro_theme_score(z_tema, PRIORS_M5).reindex(columns=cierres.columns)
        return sector_score(macro, price_momentum_score(cierres), peso_macro)

    return metrica


METRICAS["sector-score-h1"] = _sector_score(0.0)
METRICAS["sector-score-h2"] = _sector_score(0.5)


def señales_transversales(cierres, metrica, fraccion=1 / 3):
    """Panel 0/1 fecha × activo: 1 = el activo está en el tercio superior vigente; se mantiene desde la apertura siguiente.

    La decisión se toma solo en la última sesión de cada mes (calendario NYSE). Ese día se ordenan los activos
    con `metrica` disponible y se eligen los `k = max(1, N_disponibles // 3)` de mayor valor (desempate por
    ticker, alfabético). Con menos de 3 disponibles no hay pierna ese mes. La fila de la sesión de decisión
    ya trae el estado nuevo y se repite hacia adelante hasta la próxima decisión; antes de la primera, 0.

    Límite: el control de la etapa 2 contra `backtesting.py` ejecuta esta misma serie en los dos motores, así que
    NO valida que el ranking esté bien calculado. Eso lo cubren los tests de este módulo (valores conocidos,
    truncamiento sobre el panel y una implementación independiente por bucle).
    """
    valores = metrica(cierres)
    estado = pd.DataFrame(float("nan"), index=cierres.index, columns=cierres.columns)
    for fecha in cierres.index:
        if not is_last_session_of_month(fecha):
            continue
        disponibles = valores.loc[fecha].dropna()
        elegidos = []
        if len(disponibles) >= 3:
            k = max(1, int(len(disponibles) * fraccion + 1e-9))
            orden = sorted(disponibles.index, key=lambda activo: (-disponibles[activo], activo))
            elegidos = orden[:k]
        estado.loc[fecha] = 0.0
        estado.loc[fecha, elegidos] = 1.0
    return estado.ffill().fillna(0.0).astype(int)
