import pandas as pd

from quantagent.fabrica.cartera import pesos_f1
from quantagent.strategy.rebalanceo_fin_de_mes_strategy import RebalanceoFinDeMesStrategy
from quantagent.strategy.registry import build_strategy, cli_aliases


def test_alias_y_posiciones_objetivo_f1():
    idx = pd.bdate_range("2024-01-31", periods=12)
    cierres = pd.DataFrame(
        {"SPY": [100, 90, 110, 105, 100, 95, 100, 102, 104, 106, 108, 110],
         "IEF": [100, 100, 100, 100, 100, 100, 101, 102, 101, 103, 102, 104]},
        index=idx,
    )
    estrategia = build_strategy("rebalanceo-fin-de-mes")

    assert cli_aliases()["rebalanceo-fin-de-mes"] == "RebalanceoFinDeMesStrategy"
    assert isinstance(estrategia, RebalanceoFinDeMesStrategy)
    pd.testing.assert_frame_equal(estrategia.posiciones_objetivo(cierres), pesos_f1(cierres))


def test_fuera_de_ventana_curva_f1_iguala_benchmark_50_50():
    idx = pd.bdate_range("2024-01-31", periods=12)
    cierres = pd.DataFrame(
        {"SPY": [100, 90, 110, 105, 100, 95, 100, 102, 104, 106, 108, 110],
         "IEF": [100, 100, 100, 100, 100, 100, 101, 102, 101, 103, 102, 104]},
        index=idx,
    )
    pesos = RebalanceoFinDeMesStrategy.posiciones_objetivo(cierres)
    retornos = cierres.pct_change().fillna(0.0)
    fuera = (pesos == 0.5).all(axis=1)
    retorno_f1 = (retornos * pesos).sum(axis=1)
    retorno_benchmark = retornos.mean(axis=1)

    pd.testing.assert_series_equal(retorno_f1.loc[fuera], retorno_benchmark.loc[fuera])
