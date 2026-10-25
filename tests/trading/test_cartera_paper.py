from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from quantagent.trading.cartera_paper import (
    entradas_cartera,
    exposicion_piernas,
    sesiones_cerradas_nyse,
    ultima_sesion_cerrada_nyse,
)


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
