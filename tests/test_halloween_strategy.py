"""Tests de HalloweenStrategy (QuantAgent-d4cq.P04): velas sintéticas de varios años, sin red."""

from pathlib import Path

import pandas as pd

from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.strategy import build_strategy
from quantagent.strategy.halloween_strategy import HalloweenStrategy
from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases


def _velas(start="2019-01-02", end="2023-12-29") -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    close = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)
    return pd.DataFrame(
        {"timestamp": idx, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=idx,
    )


def _posicion_larga(velas: pd.DataFrame) -> pd.Series:
    sig = bar_signals(HalloweenStrategy(), velas, "SPY", "1d")
    # Estado tras cada cierre: entra con la señal, sale con la regla de salida (Backtest actúa igual).
    estado, dentro = [], False
    for entry, exit_ in zip(sig["entry"], sig["exit"]):
        if exit_:
            dentro = False
        elif entry == 1:
            dentro = True
        estado.append(dentro)
    return pd.Series(estado, index=velas.index)


def test_larga_solo_de_fin_de_octubre_a_fin_de_abril():
    velas = _velas()
    sig = bar_signals(HalloweenStrategy(), velas, "SPY", "1d")
    # Últimas sesiones de octubre (incluye 2021-10-29 viernes y 2022-10-31 lunes) y de abril (2022-04-29, 2020-04-30).
    entradas = [d.strftime("%Y-%m-%d") for d in sig.index[sig["entry"] == 1]]
    salidas = [d.strftime("%Y-%m-%d") for d in sig.index[sig["exit"]]]
    assert entradas == ["2019-10-31", "2020-10-30", "2021-10-29", "2022-10-31", "2023-10-31"]
    assert salidas == ["2019-04-30", "2020-04-30", "2021-04-30", "2022-04-29", "2023-04-28"]

    dentro = _posicion_larga(velas)
    # Entre noviembre y abril (después de la entrada) está dentro; de mayo a octubre, fuera.
    assert dentro["2021-11-01":"2022-04-28"].all()
    assert not dentro["2022-05-02":"2022-10-28"].any()
    # El cambio de estado ocurre en la sesión de fin de mes, no antes.
    assert not dentro["2022-10-28"] and dentro["2022-10-31"]
    assert dentro["2022-04-28"] and not dentro["2022-04-29"]


def test_no_mira_el_futuro_y_depende_solo_de_la_fecha():
    velas = _velas()
    cortes = [velas.index[400], velas.index[777], velas.index[1000]]
    fn = lambda c: bar_signals(HalloweenStrategy(), c, "SPY", "1d")  # noqa: E731
    assert check_no_lookahead(fn, velas, cortes) == []
    # Los precios no cambian entrada ni salida: misma señal con una serie plana o con una a la baja.
    otra = velas.assign(close=100.0, open=100.0, high=101.0, low=99.0)
    pd.testing.assert_frame_equal(fn(velas)[["entry", "exit"]], fn(otra)[["entry", "exit"]])


def test_registrada_con_alias_y_cero_parametros():
    assert cli_aliases()["halloween"] == "HalloweenStrategy"
    assert STRATEGY_REGISTRY["HalloweenStrategy"]["params"] == {}
    assert isinstance(build_strategy("HalloweenStrategy"), HalloweenStrategy)


def test_admitir_acepta_el_yaml_de_prueba():
    h = cargar_hipotesis(Path(__file__).parent / "fixtures" / "hipotesis" / "halloween-valida.yaml")
    assert admitir(h, _velas("2010-01-04", "2013-12-31")) == (True, [])
