"""QuantAgent-d4cq.13: filtrar_activo elige por ventana en el ajuste y mide en la prueba recortada."""

import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica import filtro
from quantagent.fabrica.hipotesis import Hipotesis

COSTOS = {"commission_pct": 0.0, "slippage_pct": 0.0}


@pytest.mark.parametrize("valores,elegido", [([1, 2, 1], 1), ([1, 2], 2), ([1, 1, 2], 1)])
def test_parametros_elegidos(valores, elegido):
    ganadores = [{"prueba_desde": pd.Timestamp(2010 + i, 1, 1),
                  "params": {"fast": valor, "slow": 20}} for i, valor in enumerate(valores)]
    assert filtro.parametros_elegidos(ganadores) == {"fast": elegido, "slow": 20}
    assert filtro.parametros_elegidos(ganadores[::-1]) == {"fast": elegido, "slow": 20}


@pytest.mark.parametrize("sharpe,trades,esperado,motivo", [
    (0.10, 30, "PASA", "Sharpe de cartera 0.10"),
    (0.50, 60, "PASA", "Sharpe de cartera 0.50"),
    (0.50, 24, "CORTA", "menos de 30 trades en la cartera"),
    (0.09, 30, "CORTA", "Sharpe de cartera bajo 0,10"),
    (float("nan"), 30, "CORTA", "Sharpe de cartera bajo 0,10"),
    (None, 30, "CORTA", "Sharpe de cartera bajo 0,10"),
])
def test_veredicto_etapa1_cartera(sharpe, trades, esperado, motivo):
    assert filtro.veredicto_etapa1(sharpe, trades) == (esperado, motivo)



def _velas():
    n = 365 * 8
    ts = pd.date_range("2010-01-01", periods=n, freq="D")
    close = 100 + 10 * np.sin(np.arange(n) / 20.0) + np.arange(n) * 0.01
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})


def _hip():
    return Hipotesis(id="t", autor="a", justificacion="j", clase="sma-cross", parametros={"fast": [5, 200], "slow": [20, 300]},
                     activos=("SPY",), timeframe="1d")


@pytest.fixture
def espia(monkeypatch):
    """Señales sintéticas: 'bueno' (fast=5) gana siempre en ajuste; 'malo' pierde. Registra los recortes."""
    vistos = []

    def fake_signals(strategy, candles, symbol, timeframe):
        sig = pd.DataFrame({"entry": 0, "exit": False}, index=candles.index)
        if strategy.fast == 5:  # long al inicio de cada año, sale a fin de año: sube con la tendencia
            sig.loc[candles.index.dayofyear == 2, "entry"] = 1
            sig.loc[candles.index.dayofyear == 300, "exit"] = True
        return sig

    real = filtro.run_filter

    def run(signals, candles, **kw):
        vistos.append((signals.index[0], signals.index[-1], candles.index[0], candles.index[-1]))
        return real(signals, candles, **kw)

    monkeypatch.setattr(filtro, "bar_signals", fake_signals)
    monkeypatch.setattr(filtro, "run_filter", run)
    monkeypatch.setattr(filtro, "build_strategy", lambda nombre, **kw: type("S", (), kw)())
    monkeypatch.setattr(filtro, "cli_aliases", lambda: {"sma-cross": "SmaCrossStrategy"})
    return vistos


def test_el_ganador_de_cada_ventana_es_el_claramente_mejor(espia):
    r = filtro.filtrar_activo(_hip(), _velas(), costos=COSTOS, capital=10_000.0, train_years=3, test_years=1)
    assert len(r["ganadores"]) >= 2
    assert all(g["params"] == {"fast": 5, "slow": 20} for g in r["ganadores"])
    assert r["trades"] > 0 and r["sharpe_tasa_0"] != 0


def test_la_prueba_no_usa_velas_de_la_ventana_siguiente(espia):
    r = filtro.filtrar_activo(_hip(), _velas(), costos=COSTOS, capital=10_000.0, train_years=3, test_years=1)
    for g in r["ganadores"]:
        desde, hasta = pd.Timestamp(g["prueba_desde"]), pd.Timestamp(g["prueba_hasta"])
        recortes = [v for v in espia if v[0] >= desde and v[1] <= hasta and v[0].year == desde.year]
        assert recortes and all(v[0] == v[2] and v[1] == v[3] for v in recortes)  # señales y velas, mismo tramo
        assert all(v[1] <= hasta for v in recortes)  # nada posterior al fin de la prueba


def test_retornos_solo_de_pruebas_y_reproducen_el_sharpe(espia):
    r = filtro.filtrar_activo(_hip(), _velas(), costos=COSTOS, capital=10_000.0, train_years=3, test_years=1)
    ret = r["retornos"]
    assert isinstance(ret.index, pd.DatetimeIndex) and ret.index.is_unique
    pruebas = [(pd.Timestamp(g["prueba_desde"]), pd.Timestamp(g["prueba_hasta"])) for g in r["ganadores"]]
    assert all(any(d <= f <= h for d, h in pruebas) for f in ret.index)
    assert ret.index.min() >= pruebas[0][0] and ret.index.max() <= pruebas[-1][1]
    equity = 10_000.0 * (1 + ret).cumprod()
    assert filtro.equity_metrics(equity, 0.0)["sharpe"] == pytest.approx(r["sharpe_tasa_0"])
