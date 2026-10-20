"""Tests de TurnOfTheMonthStrategy (QuantAgent-rpna)."""

import pandas as pd
from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.strategy import build_strategy
from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases
from quantagent.strategy.turn_of_the_month_strategy import TurnOfTheMonthStrategy


def _velas(fechas: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(fechas)
    return pd.DataFrame({"timestamp": idx, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=idx)


def test_fechas_sinteticas_con_feriados_abre_y_cierra_en_dias_habiles_exactos():
    fechas = ["2020-01-30", "2020-01-31", "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-03-30", "2020-03-31", "2020-10-29", "2020-10-30"]
    sig = bar_signals(TurnOfTheMonthStrategy(), _velas(fechas), "SPY", "1d")
    assert [d.strftime("%Y-%m-%d") for d in sig.index[sig["entry"] == 1]] == ["2020-01-31", "2020-03-31", "2020-10-30"]
    assert [d.strftime("%Y-%m-%d") for d in sig.index[sig["exit"]]] == ["2020-02-06"]


def test_la_senal_de_un_dia_no_cambia_al_agregar_velas_posteriores():
    v_base = _velas([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-01-04", "2021-06-30")])
    v_ext = _velas([d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-01-04", "2021-12-31")])
    s_base, s_ext = bar_signals(TurnOfTheMonthStrategy(), v_base, "SPY", "1d"), bar_signals(TurnOfTheMonthStrategy(), v_ext, "SPY", "1d")
    pd.testing.assert_frame_equal(s_base[["entry", "exit"]], s_ext.loc[s_base.index, ["entry", "exit"]])
    cortes = [v_base.index[30], v_base.index[60], v_base.index[90]]
    assert check_no_lookahead(lambda c: bar_signals(TurnOfTheMonthStrategy(), c, "SPY", "1d"), v_base, cortes) == []


def test_registrada_con_alias_y_cero_parametros():
    assert cli_aliases()["turn-of-the-month"] == "TurnOfTheMonthStrategy"
    assert STRATEGY_REGISTRY["TurnOfTheMonthStrategy"]["params"] == {}
    assert isinstance(build_strategy("TurnOfTheMonthStrategy"), TurnOfTheMonthStrategy)


def test_viernes_santo_en_fin_de_mes_abre_jueves_y_no_abre_viernes_santo():
    fechas = ["2018-03-28", "2018-03-29", "2018-03-30", "2018-04-02", "2018-04-03", "2018-04-04", "2018-04-05", "2024-03-27", "2024-03-28", "2024-03-29", "2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"]
    sig = bar_signals(TurnOfTheMonthStrategy(), _velas(fechas), "SPY", "1d")
    assert [d.strftime("%Y-%m-%d") for d in sig.index[sig["entry"] == 1]] == ["2018-03-29", "2024-03-28"]
    assert "2018-03-30" not in sig.index[sig["entry"] == 1] and "2024-03-29" not in sig.index[sig["entry"] == 1]
    assert [d.strftime("%Y-%m-%d") for d in sig.index[sig["exit"]]] == ["2018-04-04", "2024-04-03"]
