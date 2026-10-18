"""Piernas transversales: ranking de activos del universo, un tercio por arriba (QuantAgent P05.1)."""
import pandas as pd

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


def baja_volatilidad(cierres):
    """-σ de los retornos diarios de las últimas 252 sesiones (mayor = más calmo). NaN sin 252 retornos completos."""
    retornos = cierres.pct_change(fill_method=None)
    return -retornos.rolling(VENTANA_VOLATILIDAD, min_periods=VENTANA_VOLATILIDAD).std()


METRICAS["baja-volatilidad"] = baja_volatilidad


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
