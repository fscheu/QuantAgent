from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from quantagent.trading.cartera_paper import (
    entradas_cartera,
    exposicion_piernas,
    pesos_operables,
    sesiones_cerradas_nyse,
    ultima_sesion_cerrada_nyse,
)
from quantagent.fabrica.cartera import pesos, pesos_objetivo
from quantagent.fabrica.hipotesis import cargar_hipotesis


def test_horario_regular_corte_sesion():
    # 2024-05-15 es miércoles regular (EDT: cierre 20:00 UTC)
    antes = datetime(2024, 5, 15, 19, 59, tzinfo=timezone.utc)
    despues = datetime(2024, 5, 15, 20, 1, tzinfo=timezone.utc)
    assert ultima_sesion_cerrada_nyse(antes) == pd.Timestamp("2024-05-14")
    assert ultima_sesion_cerrada_nyse(despues) == pd.Timestamp("2024-05-15")


def test_cierre_temprano_black_friday_sin_corte_16h():
    # 2023-11-24 Black Friday cierra 13:00 EST (18:00 UTC); jueves 23 fue feriado
    antes = pd.Timestamp("2023-11-24 17:59:00", tz="UTC")
    en_cierre = pd.Timestamp("2023-11-24 18:00:00", tz="UTC")
    entre_13h_y_16h_est = pd.Timestamp("2023-11-24 19:30:00", tz="UTC")
    assert ultima_sesion_cerrada_nyse(antes) == pd.Timestamp("2023-11-22")
    assert ultima_sesion_cerrada_nyse(en_cierre) == pd.Timestamp("2023-11-24")
    assert ultima_sesion_cerrada_nyse(entre_13h_y_16h_est) == pd.Timestamp("2023-11-24")


def test_fin_de_semana_y_feriado_conservan_sesion():
    # Memorial Day 2023: viernes 26 mayo cierra 20:00 UTC; lunes 29 feriado
    viernes_post = pd.Timestamp("2023-05-26 21:00:00", tz="UTC")
    sabado = pd.Timestamp("2023-05-27 14:00:00", tz="UTC")
    lunes_feriado = pd.Timestamp("2023-05-29 18:00:00", tz="UTC")
    esperada = pd.Timestamp("2023-05-26")
    assert ultima_sesion_cerrada_nyse(viernes_post) == esperada
    assert ultima_sesion_cerrada_nyse(sabado) == esperada
    assert ultima_sesion_cerrada_nyse(lunes_feriado) == esperada


def test_ventana_sesiones_sin_futuro_y_reloj_inyectable():
    as_of_naive = datetime(2024, 5, 15, 20, 30)
    as_of_aware = pd.Timestamp("2024-05-15 20:30:00", tz="UTC")
    s_naive = sesiones_cerradas_nyse(as_of_naive, n_sesiones=253)
    s_aware = sesiones_cerradas_nyse(as_of_aware, n_sesiones=253)
    assert s_naive.equals(s_aware)
    assert len(s_naive) == 253
    assert s_naive.is_monotonic_increasing
    assert s_naive[-1] == pd.Timestamp("2024-05-15")
    assert (s_naive <= pd.Timestamp("2024-05-15")).all()


class FakeProvider:
    def __init__(self, dfs):
        self.dfs = dfs
        self.calls = []

    def get_ohlc(self, s, tf, start, end):
        assert start < end
        self.calls.append((s, start, end))
        df = self.dfs.get(s, pd.DataFrame(columns=["timestamp", "close"]))
        return df[(df["timestamp"] >= start) & (df["timestamp"] < end)]


@pytest.fixture
def market():
    as_of = pd.Timestamp("2024-05-16 05:00:00", tz="UTC")
    ses = sesiones_cerradas_nyse(as_of, n_sesiones=300)
    h = SimpleNamespace(piernas=[SimpleNamespace(activos=("A", "B"))])
    return as_of, ses, pd.DataFrame({"timestamp": ses, "close": 100.0}), h


def test_entradas_cartera_panel_completo_y_held_extra_una_barra(market):
    as_of, ses, df, h = market
    df_c = pd.DataFrame({"timestamp": [ses[-1]], "close": 50.0})
    prov = FakeProvider({"A": df, "B": df, "C": df_c})
    res = entradas_cartera(h, prov, lambda: as_of, {"C": {"qty": 10}})
    assert res is not None
    ultima, panel, precios = res
    assert ultima == pd.Timestamp("2024-05-15")
    assert panel.shape == (300, 2)
    assert precios == {"A": 100.0, "B": 100.0, "C": 50.0}
    assert ("C", pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-16")) in prov.calls


@pytest.mark.parametrize(
    "mod_a,held",
    [
        (lambda df: df.iloc[-200:], []),
        (lambda df: df.drop(df.index[-50]), []),
        (lambda df: df.iloc[:-1], []),
        (lambda df: df, ["C"]),
    ],
)
def test_entradas_cartera_casos_que_impiden_operar(market, mod_a, held):
    as_of, _, df, h = market
    prov = FakeProvider({"A": mod_a(df), "B": df})
    assert entradas_cartera(h, prov, lambda: as_of, held) is None


def test_entradas_cartera_excluye_barras_intradia_y_futuras(market):
    _, _, df, h = market
    futuro = pd.DataFrame({"timestamp": [pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-16")], "close": 999.0})
    prov = FakeProvider({"A": pd.concat([df, futuro]), "B": df})
    res = entradas_cartera(h, prov, lambda: pd.Timestamp("2024-05-15 14:00:00", tz="UTC"))
    assert res is not None
    ultima, panel, precios = res
    assert ultima == pd.Timestamp("2024-05-14")
    assert panel.index[-1] == pd.Timestamp("2024-05-14")
    assert precios["A"] == 100.0


def test_exposicion_piernas_momentum_y_sma_cross():
    idx = pd.bdate_range("2023-01-01", "2023-04-28")
    panel = pd.DataFrame({"A": [100.0 + i for i in range(len(idx))], "B": [100.0 - i for i in range(50)] + [50.0 + i * 2 for i in range(len(idx) - 50)]}, index=idx)
    h = SimpleNamespace(timeframe="1d", piernas=[
        SimpleNamespace(clase="momentum-12m", activos=("A",), parametros={"lookback": 40}),
        SimpleNamespace(clase="sma-cross", activos=("B",), parametros={"fast": 5, "slow": 15}),
    ])
    expos = exposicion_piernas(h, panel)
    assert (expos[(0, "A")].loc[:"2023-02-27"] == 0).all() and expos[(0, "A")].loc["2023-02-28"] == 1
    assert (expos[(0, "A")].loc["2023-03-01":"2023-03-31"] == 1).all()
    assert expos[(1, "B")].iloc[55] == 1 and expos[(1, "B")].iloc[-1] == 1


def test_exposicion_piernas_transversal_y_activos_compartidos():
    idx = pd.bdate_range("2022-01-01", "2023-03-31")
    panel = pd.DataFrame({a: [100.0 + i * (j + 1) for i in range(len(idx))] for j, a in enumerate(list("ABCDEF"))}, index=idx)
    h = SimpleNamespace(timeframe="1d", piernas=[
        SimpleNamespace(clase="momentum-12m", activos=("A", "F"), parametros={"lookback": 20}),
        SimpleNamespace(clase="momentum-transversal", activos=tuple("ABCDEF"), parametros={}),
    ])
    expos = exposicion_piernas(h, panel)
    assert (0, "A") in expos.columns and (1, "A") in expos.columns and (0, "F") in expos.columns and (1, "F") in expos.columns
    assert (expos[(1, "F")].loc["2023-01-31":"2023-02-27"] == 1).all()
    assert (expos[(1, "A")].loc["2023-01-31":"2023-02-27"] == 0).all()


def test_exposicion_piernas_sin_mirar_futuro():
    from quantagent.backtesting.signals import check_no_lookahead

    idx = pd.bdate_range("2021-01-01", periods=300)
    panel = pd.DataFrame({s: [100.0 + i for i in range(300)] for s in list("ABCDEF")}, index=idx)
    h = SimpleNamespace(timeframe="1d", piernas=[
        SimpleNamespace(clase="momentum-12m", activos=("A",), parametros={"lookback": 20}),
        SimpleNamespace(clase="sma-cross", activos=("B",), parametros={"fast": 3, "slow": 8}),
        SimpleNamespace(clase="momentum-transversal", activos=tuple("ABCDEF"), parametros={}),
    ])
    assert check_no_lookahead(lambda p: exposicion_piernas(h, p), panel, [panel.index[260], panel.index[280]]) == []


def test_curva_pendiente_usa_macro_publicada_y_marca_bloqueo():
    from quantagent.data.macro_vivo import MacroVencida, serie_vigente

    sesiones = pd.DatetimeIndex(["2024-01-08", "2024-01-09"])
    class Cliente:
        def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
            return pd.DataFrame({"fecha": pd.to_datetime(["2024-01-05", "2024-01-08"]), "valor": [0.25, 0.0],
                                 "realtime_start": pd.to_datetime(["2024-01-05", "2024-01-08"]),
                                 "realtime_end": pd.to_datetime(["2262-04-11", "2262-04-11"])})
    assert serie_vigente("T10Y2Y", sesiones, Cliente()).tolist() == [0.25, 0.0]
    panel = pd.DataFrame({"TLT": [100.0, 101.0]}, index=sesiones)
    h = SimpleNamespace(piernas=[SimpleNamespace(clase="curva-pendiente", activos=("TLT",), parametros={})])
    expos = exposicion_piernas(h, panel, Cliente())
    assert expos[(0, "TLT")].tolist() == [1, 0] and expos.attrs["bloqueos"] == []
    class Vencido(Cliente):
        def observaciones(self, *args, **kwargs):
            return pd.DataFrame({"fecha": pd.to_datetime(["2024-01-01"]), "valor": [0.25],
                                 "realtime_start": pd.to_datetime(["2024-01-01"]),
                                 "realtime_end": pd.to_datetime(["2262-04-11"])})
    with pytest.raises(MacroVencida, match="vencida"):
        serie_vigente("T10Y2Y", sesiones, Vencido())
    bloqueada = exposicion_piernas(h, panel, Vencido())
    assert bloqueada[(0, "TLT")].isna().all() and "vencida" in bloqueada.attrs["bloqueos"][0]
    operables = pesos_operables(h, panel, sesiones[-1], Vencido())
    assert operables["TLT"] == 0 and "vencida" in operables.attrs["bloqueos"][0]
    class SinDatos(Cliente):
        def observaciones(self, *args, **kwargs):
            return pd.DataFrame(columns=["fecha", "valor", "realtime_start", "realtime_end"])
    sin_macro = exposicion_piernas(h, panel, SinDatos())
    assert sin_macro[(0, "TLT")].isna().all() and "sin observación" in sin_macro.attrs["bloqueos"][0]


def test_pesos_operables_compone_exposicion_y_carga_ambas_yaml():
    from pathlib import Path
    from quantagent.fabrica.transversal import es_transversal

    raiz = Path(__file__).resolve().parents[2]
    macro = pd.Series(0.5, index=pd.bdate_range("2023-01-02", periods=300))
    for nombre in ("cartera-tendencia-momentum", "cartera-nucleo-curva"):
        ruta = raiz / "hipotesis" / f"{nombre}.yaml"
        h = cargar_hipotesis(ruta)
        idx = macro.index
        simbolos = sorted({a for p in h.piernas for a in p.activos})
        panel = pd.DataFrame({a: 100 + j + pd.Series(range(len(idx)), index=idx) for j, a in enumerate(simbolos)}, index=idx)
        expos = exposicion_piernas(h, panel, macro)
        seleccion = {i: expos.xs(i, axis=1, level=0) for i, p in enumerate(h.piernas) if es_transversal(p.clase)}
        referencia = pesos(panel, {i: p.activos for i, p in enumerate(h.piernas)}, seleccion=seleccion).loc[idx[-1]]
        esperado = {}
        for (i, activo), peso in referencia.items():
            esperado[activo] = esperado.get(activo, 0.0) + peso * expos.loc[idx[-1], (i, activo)]
        resultado = pesos_operables(ruta, panel, idx[-1], macro)
        assert resultado.to_dict() == pytest.approx(esperado)
        assert resultado.sum() <= sum(pesos_objetivo(panel, {i: p.activos for i, p in enumerate(h.piernas)}, idx[-1]).values())


def test_pesos_operables_suma_simbolo_y_conserva_efectivo_y_calendario(monkeypatch):
    from types import SimpleNamespace
    import quantagent.trading.cartera_paper as cartera_paper

    idx = pd.bdate_range("2024-01-02", periods=40)
    panel = pd.DataFrame(100.0, index=idx, columns=list("ABC"))
    h = SimpleNamespace(piernas=[SimpleNamespace(clase="momentum-12m", activos=tuple("AB")),
                                 SimpleNamespace(clase="momentum-12m", activos=tuple("AC"))])
    expos = pd.DataFrame(0, index=idx, columns=pd.MultiIndex.from_tuples([(0, a) for a in "AB"] + [(1, a) for a in "AC"]))
    expos[(0, "A")] = expos[(1, "A")] = 1
    expos.attrs["bloqueos"] = []
    monkeypatch.setattr(cartera_paper, "exposicion_piernas", lambda *args: expos)

    enero = pesos_operables(h, panel, idx[0])
    fin_de_enero = pesos_operables(h, panel, idx[20])
    assert enero.to_dict() == pytest.approx({"A": 0.5, "B": 0.0, "C": 0.0})
    assert fin_de_enero.to_dict() == pytest.approx(enero.to_dict())
    assert enero.sum() == pytest.approx(0.5)
