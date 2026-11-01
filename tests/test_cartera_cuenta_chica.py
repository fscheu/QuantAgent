import pandas as pd
import pytest
import numpy as np

from quantagent.backtesting.cartera_cuenta_chica import simular_cartera_cuenta_chica
from quantagent.trading.rebalanceador import ordenes_rebalanceo
from quantagent.models import OrderSide


class _Hypothesis:
    class _Leg:
        activos = ("AAA", "BBB")
        clase = "rsi"
    piernas = (_Leg(),)


def test_rebalanceador_floor_y_venta_precede_compra():
    orders = ordenes_rebalanceo({"AAA": .5, "BBB": .5}, {"AAA": {"qty": 4}},
                                {"AAA": 10, "BBB": 10}, 50)
    assert [(side, qty) for _, side, qty, _ in orders] == [(OrderSide.SELL, 2), (OrderSide.BUY, 2)]


def test_simulacion_reporta_activo_que_no_entra_con_capital_chico(monkeypatch):
    from quantagent.backtesting import cartera_cuenta_chica as module
    panel = pd.DataFrame({"AAA": [100., 100.], "BBB": [100., 100.]},
                         index=pd.to_datetime(["2018-01-02", "2018-01-03"], utc=True))
    monkeypatch.setattr(module, "pesos_operables_por_fecha", lambda *_, **__: pd.DataFrame(
        [{"AAA": .5, "BBB": .5}] * len(panel), index=panel.index))
    result = simular_cartera_cuenta_chica(_Hypothesis(), panel, 100)
    assert result["zero_positions"] == ["AAA", "BBB"]
    assert result["equity_curve"][-1]["equity"] == pytest.approx(100)


def _hipotesis():
    class Leg:
        clase = "sma-cross"
        parametros = {"fast": [2], "slow": [3]}
        activos = ("AAA", "BBB")

        def combinaciones(self):
            return [{"fast": 2, "slow": 3}]

    class Hypothesis:
        piernas = (Leg(),)
        timeframe = "1d"

    return Hypothesis()


def _panel():
    idx = pd.bdate_range("2017-10-02", periods=65, tz="UTC")
    a = np.r_[np.linspace(100, 60, 43), np.linspace(100, 120, 22)]
    b = np.linspace(30, 20, len(idx))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def test_sma_cross_rebalance_costos_y_equity_manual():
    panel = _panel()
    result = simular_cartera_cuenta_chica(_hipotesis(), panel, 1000,
                                          slippage_pct={"AAA": .01, "BBB": .01},
                                          commission_pct=.005)
    entry = panel.iloc[44]["AAA"]
    target_qty = int(1000 * .95 * .5 / entry)
    fee = target_qty * entry * .015
    cash = 1000 - target_qty * entry - fee
    expected = cash + target_qty * panel.iloc[-1]["AAA"]
    assert result["equity_curve"][-1]["equity"] == pytest.approx(expected)
    assert result["costs"] == pytest.approx(fee)


def test_posiciones_en_cero_depende_del_capital():
    panel = _panel() * 6
    low = simular_cartera_cuenta_chica(_hipotesis(), panel, 1000)
    high = simular_cartera_cuenta_chica(_hipotesis(), panel, 5000)
    assert "AAA" in low["zero_positions"]
    assert "AAA" not in high["zero_positions"]


def test_sin_costos_capital_grande_se_aproxima_a_curva_cartera():
    from quantagent.fabrica.cartera import curva_cartera
    from quantagent.trading.cartera_paper import exposicion_piernas

    panel = _panel()
    h = _hipotesis()
    result = simular_cartera_cuenta_chica(h, panel, 1_000_000_000)
    retornos = pd.DataFrame({(0, s): panel[s].pct_change().fillna(0) for s in panel})
    h_piernas = {0: ("AAA", "BBB")}
    expos = exposicion_piernas(h, panel)
    seleccion = {0: expos.xs(0, axis=1, level=0).shift(1).fillna(0)}
    # Una señal observada al cierre solo afecta el retorno de la rueda siguiente.
    referencia = curva_cartera(panel, h_piernas, retornos,
                               capital=1_000_000_000, seleccion=seleccion)
    # La diferencia restante viene de la reserva del 5%, lotes enteros y costos nulos.
    assert result["equity_curve"][-1]["equity"] / 1_000_000_000 == pytest.approx(
        referencia.iloc[-1] / 1_000_000_000, rel=.10)


def test_pesos_precalculados_coinciden_con_paneles_causales(monkeypatch):
    from quantagent.backtesting.cartera_cuenta_chica import pesos_operables_por_fecha
    from quantagent.trading.cartera_paper import pesos_operables
    from quantagent.fabrica import transversal

    class PiernaTransversal:
        clase = "test-transversal"
        activos = tuple(f"T{i}" for i in range(6))

    class Hipotesis:
        piernas = (_hipotesis().piernas[0], PiernaTransversal())
        timeframe = "1d"

    monkeypatch.setitem(transversal.METRICAS, "test-transversal", lambda cierres: cierres)
    idx = pd.bdate_range("2018-01-02", periods=90, tz="UTC")
    panel = pd.DataFrame({"AAA": 100 + np.arange(len(idx)) * np.where(np.arange(len(idx)) < 45, -1, 1),
                          "BBB": 120 - np.arange(len(idx)) * .3,
                          **{f"T{i}": 80 + np.arange(len(idx)) * (i + 1) / 100 for i in range(6)}}, index=idx)
    h = Hipotesis()
    precalculados = pesos_operables_por_fecha(h, panel)
    for fecha in panel.index:
        esperado = pesos_operables(h, panel.loc[:fecha], fecha)
        pd.testing.assert_series_equal(precalculados.loc[fecha].reindex(precalculados.columns).fillna(0),
                                       esperado.reindex(precalculados.columns).fillna(0),
                                       check_names=False, check_dtype=False)
