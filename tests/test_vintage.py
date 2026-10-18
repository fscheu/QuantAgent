"""QuantAgent-d4cq.71 (M03): la prueba de truncamiento corta también las versiones macro, no solo las velas."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quantagent.backtesting.signals import bar_signals, check_no_lookahead_vintage
from quantagent.data.macro import recortar_vintage, valor_conocido_en
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.strategy.registry import STRATEGY_REGISTRY

FIN = pd.Timestamp("9999-12-31")
T = pd.Timestamp


def _versiones(filas):
    return pd.DataFrame(filas, columns=["fecha", "valor", "realtime_start", "realtime_end"]).assign(
        fecha=lambda d: pd.to_datetime(d["fecha"]),
        realtime_start=lambda d: pd.to_datetime(d["realtime_start"]),
        realtime_end=lambda d: pd.to_datetime(d["realtime_end"]),
    )


# Abril se publica el 10/4 con 4.0 y el 15/5 se revisa a 4.5. Umbral de las estrategias de prueba: 4.2.
MACRO = {"X": _versiones([("2020-04-01", 4.0, "2020-04-10", "2020-05-14"), ("2020-04-01", 4.5, "2020-05-15", FIN)])}
UMBRAL = 4.2


def _velas(desde="2020-04-01", n=60):
    idx = pd.bdate_range(desde, periods=n)
    return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}, index=idx)


class _Base(TradingStrategy):
    """Largo cuando `_valor(fecha)` supera el umbral. Subclases: de dónde sale el valor."""

    requiere_macro = ("X",)

    def __init__(self, macro):
        self.macro = macro

    def _valor(self, fecha):
        raise NotImplementedError

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        valor = self._valor(pd.Timestamp(kline_data[-1]["timestamp"]))
        if valor is None or valor <= UMBRAL:
            return None
        return TradingSignal(decision="LONG", confidence=1.0, entry_price=current_price)

    def should_reevaluate(self, position, current_price):
        return False


class Conocido(_Base):
    """Usa lo publicado en la fecha: `valor_conocido_en`."""

    def _valor(self, fecha):
        k = valor_conocido_en(self.macro["X"], fecha)
        return None if k is None else k[1]


class UltimaRevision(_Base):
    """Hace trampa: toma la versión más nueva que tenga el snapshot de cada observación ya ocurrida."""

    def _valor(self, fecha):
        df = self.macro["X"]
        df = df[df["fecha"] <= fecha]
        return None if df.empty else df.sort_values("realtime_start")["valor"].iloc[-1]


def _fn(cls):
    return lambda velas, macro: bar_signals(cls(macro), velas, "SPY", "1d")


CORTES = [T("2020-04-30"), T("2020-05-29")]


def test_recortar_descarta_versiones_futuras_y_acota_realtime_end():
    r = recortar_vintage(MACRO, "2020-05-01")["X"]
    assert r["valor"].tolist() == [4.0]
    assert r["realtime_end"].tolist() == [T("2020-05-01")]
    assert MACRO["X"]["realtime_end"].iloc[1] == FIN  # no toca el original


def test_recorte_incluye_el_borde_realtime_start_igual_a_T():
    assert recortar_vintage(MACRO, "2020-05-15")["X"]["valor"].tolist() == [4.0, 4.5]
    assert recortar_vintage(MACRO, "2020-05-14")["X"]["valor"].tolist() == [4.0]


def test_valor_conocido_pasa_la_prueba_y_ve_la_revision_recien_el_dia_que_sale():
    velas = _velas()
    assert check_no_lookahead_vintage(_fn(Conocido), velas, MACRO, CORTES) == []
    entry = _fn(Conocido)(velas, MACRO)["entry"]
    assert (entry[entry.index < "2020-05-15"] == 0).all()  # con 4.0 no hay señal
    assert (entry[entry.index >= "2020-05-15"] == 1).all()  # con 4.5 sí


def test_ultima_revision_falla_a_proposito():
    difs = check_no_lookahead_vintage(_fn(UltimaRevision), _velas(), MACRO, CORTES)
    assert difs and difs[0].startswith("entry:") and "2020-04-10" in difs[0]


def test_la_prueba_de_velas_sola_no_ve_la_trampa():
    """Por qué existe M03: cortar solo las velas deja pasar a la estrategia tramposa."""
    from quantagent.backtesting.signals import check_no_lookahead

    assert check_no_lookahead(lambda c: _fn(UltimaRevision)(c, MACRO), _velas(), CORTES) == []


# Serie diaria: la observación del viernes 6/3 se registra el lunes 9/3 (demora de publicación).
DIARIA = {"D": _versiones([("2020-03-05", 1.0, "2020-03-06", FIN), ("2020-03-06", 2.0, "2020-03-09", FIN)])}


def test_fin_de_semana_se_conoce_el_jueves_hasta_que_se_registra_el_viernes():
    d = DIARIA["D"]
    assert valor_conocido_en(d, "2020-03-06") == (T("2020-03-05"), 1.0)
    assert valor_conocido_en(d, "2020-03-07") == (T("2020-03-05"), 1.0)  # sábado
    assert valor_conocido_en(d, "2020-03-08") == (T("2020-03-05"), 1.0)  # domingo
    assert valor_conocido_en(d, "2020-03-09") == (T("2020-03-06"), 2.0)  # lunes
    assert recortar_vintage(DIARIA, "2020-03-08")["D"]["valor"].tolist() == [1.0]


def test_leer_la_observacion_del_mismo_dia_se_detecta_en_serie_diaria():
    class MismoDia(_Base):
        requiere_macro = ("D",)

        def _valor(self, fecha):
            df = self.macro["D"]
            df = df[df["fecha"] <= fecha]
            return None if df.empty else df["valor"].iloc[-1] + UMBRAL  # siempre sobre el umbral

    velas = _velas("2020-03-02", 10)
    cortes = [T("2020-03-06")]
    assert check_no_lookahead_vintage(lambda c, m: bar_signals(MismoDia(m), c, "SPY", "1d"), velas, DIARIA, cortes) != []

    class Publicado(MismoDia):
        def _valor(self, fecha):
            k = valor_conocido_en(self.macro["D"], fecha)
            return None if k is None else k[1] + UMBRAL

    assert check_no_lookahead_vintage(lambda c, m: bar_signals(Publicado(m), c, "SPY", "1d"), velas, DIARIA, cortes) == []


H = cargar_hipotesis(Path(__file__).parent / "fixtures" / "hipotesis" / "rsi-valida.yaml")


def _hipotesis(nombre):
    return replace(H, piernas=(replace(H.piernas[0], clase=nombre, parametros={}),))


@pytest.fixture
def registradas(monkeypatch):
    for cls in (Conocido, UltimaRevision):
        monkeypatch.setitem(
            STRATEGY_REGISTRY, cls.__name__,
            {"cls": cls, "type": "deterministic", "cli_alias": None, "display_name": cls.__name__, "description": "", "min_bars": 1, "params": {}},
        )


def test_admitir_acepta_la_clase_causal_y_rechaza_la_tramposa(registradas):
    velas = _velas()
    assert admitir(_hipotesis("Conocido"), velas, macro=MACRO) == (True, [])
    ok, motivos = admitir(_hipotesis("UltimaRevision"), velas, macro=MACRO)
    assert not ok and len(motivos) == 1 and "truncamiento" in motivos[0] and "entry" in motivos[0]


def test_admitir_rechaza_clase_con_macro_si_no_hay_snapshot_para_probarla(registradas):
    ok, motivos = admitir(_hipotesis("Conocido"), _velas())
    assert not ok and "requiere macro" in motivos[0]
    ok, motivos = admitir(_hipotesis("Conocido"), _velas(), macro={"OTRA": MACRO["X"]})
    assert not ok and "X" in motivos[0]
