import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.strategy.arquetipo_a_strategy import ArquetipoAStrategy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/m5"))
import h4  # noqa: E402


def test_elegibilidad_semanal_risk_off_y_limites():
    semanas = pd.to_datetime(["2011-01-03", "2011-01-10", "2011-01-18"])
    w3 = pd.DataFrame({"ACC": [0.5, 0, 0], "SPY": [0.5, 0, 1], "ETF": [0, 0, 0]}, index=semanas)
    sesiones = pd.bdate_range("2010-12-31", "2011-01-21").append(pd.DatetimeIndex(["2019-01-01"]))
    habilitados = h4.elegibilidad(w3, sesiones)
    esperado = pd.DataFrame([[True, True, False]] * 5 + [[False] * 3] * 6 + [[False, True, False]] * 4,
                           index=sesiones[1:-1], columns=w3.columns)
    pd.testing.assert_frame_equal(habilitados, esperado)
    assert not habilitados.loc["2011-01-10":"2011-01-17"].any().any()
    assert habilitados.loc["2011-01-18", "SPY"] and not habilitados.loc["2011-01-18", "ACC"]


def test_senales_calentamiento_truncamiento_y_stops():
    n = 200
    cierre = 50 + np.arange(n) * 0.5 + 3 * np.sin(np.linspace(0, 4 * np.pi, n))
    cierre[-25:] += (np.arange(25) * 0.3) ** 1.3
    cierre[-1] = cierre[-2] - 1.5
    velas = pd.DataFrame({"open": cierre - 0.5, "high": cierre + 1, "low": cierre - 1,
                          "close": cierre, "volume": 1000}, index=pd.bdate_range(end="2011-01-14", periods=n))
    futuro = velas.iloc[[-1]].set_axis(pd.DatetimeIndex(["2019-01-01"])) * 100
    fn = lambda v: h4.senales_a(v, "ACC")  # noqa: E731
    senales = fn(pd.concat([velas, futuro]))
    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / "hipotesis/arquetipo-a.yaml").read_text())
    estrategia = ArquetipoAStrategy(**{k: v[0] for k, v in spec["parametros"].items()})
    pd.testing.assert_frame_equal(senales, bar_signals(estrategia, velas, "ACC", "1d").loc["2011":], check_freq=False)
    pd.testing.assert_frame_equal(senales, fn(velas), check_freq=False)
    assert check_no_lookahead(fn, velas, [velas.index[-3]]) == []
    ultima = senales.iloc[-1]
    assert ultima["entry"] == 1 and ultima["entry_order"] == "stop" and ultima["entry_expiry_bars"] == 3
    assert ultima["entry_price"] > velas.iloc[-1]["high"] and ultima["stop_loss_pct"] > 0
    assert senales["chandelier_dist"].notna().all() and senales.index.min() >= pd.Timestamp("2011-01-01")


def test_orden_elegibilidad_cancelacion_vencimiento_y_gap():
    senal = dict(entry=1, entry_price=100, entry_expiry_bars=3, stop_loss_pct=0.1)
    baja = dict(open=90, high=95, low=85, close=91)
    gap = dict(open=105, high=110, low=100, close=108)
    assert h4.orden_a(None, gap, senal, False) == (None, None)
    pendiente, entrada = h4.orden_a(None, gap, senal, True)
    assert entrada is None  # señal al cierre: no llena en su propia vela
    assert h4.orden_a(pendiente.copy(), gap, senal, False) == (None, None)
    for restantes in (2, 1, 0):
        pendiente, entrada = h4.orden_a(pendiente, baja, senal, True)
        assert entrada is None
        assert (pendiente["restantes"] if pendiente else 0) == restantes
    assert h4.orden_a(pendiente, gap, dict(senal, entry=0), True) == (None, None)
    pendiente, _ = h4.orden_a(None, baja, senal, True)
    pendiente, _ = h4.orden_a(pendiente, baja, senal, True)
    pendiente, _ = h4.orden_a(pendiente, baja, senal, True)
    pendiente, entrada = h4.orden_a(pendiente, gap, senal, True)  # tercera vela todavía llena
    assert pendiente is None and entrada["entry_price"] == 105 and entrada["stop_loss"] == 94.5


def test_posicion_fuera_del_conjunto_stop_y_chandelier_temporal():
    senal = dict(entry=1, entry_price=100, entry_expiry_bars=3, stop_loss_pct=0.1, chandelier_dist=10)
    vela = dict(open=100, high=105, low=99, close=104)
    pendiente, _ = h4.orden_a(None, vela, senal, True)
    _, posicion = h4.orden_a(pendiente, vela, senal, True)
    for habilitado, distancia in ((True, 10), (False, 20), (False, 10)):
        assert h4.orden_a(None, vela, senal, habilitado)[1] is None
        assert h4.posicion_a(posicion, vela, dict(senal, chandelier_dist=distancia)) is None
        assert posicion["stop_loss"] == 95 and not posicion["salir"]
    assert h4.posicion_a(posicion.copy(), dict(vela, low=94), senal) == ("STOP_LOSS", 95)
    assert h4.posicion_a(posicion.copy(), dict(vela, open=90, low=88), senal) == ("STOP_LOSS", 90)
    assert h4.posicion_a(posicion, dict(open=100, high=115, low=96, close=110), senal) is None
    assert posicion["stop_loss"] == 105  # low 96 no puede tocar el stop calculado recién al cierre
    assert h4.posicion_a(posicion, dict(open=110, high=125, low=106, close=112), senal) is None
    assert posicion["salir"] and posicion["stop_loss"] == 105
    assert h4.posicion_a(posicion, dict(open=108, high=120, low=100, close=118), senal) == ("TRAILING_STOP", 108)
    _, recien_abierta = h4.orden_a(pendiente, vela, senal, True)
    assert h4.posicion_a(recien_abierta, dict(vela, low=85), senal) == ("STOP_LOSS", 90)


def test_curva_d2_fracciones_tope_costos_y_persistencia():
    fechas = pd.bdate_range("2011-01-03", periods=4)
    velas = {t: pd.DataFrame(dict(open=[100, 103, 110, 90], high=[100, 110, 115, 95],
                                low=[99, 103, 104, 85], close=[100, 110, 112, 92]), index=fechas) for t in ("R", "C")}
    senales = {t: pd.DataFrame(dict(entry=[1, 0, 0, 0], entry_price=101, entry_expiry_bars=3,
                                  stop_loss_pct=pct, chandelier_dist=100), index=fechas) for t, pct in (("R", .1), ("C", .01))}
    diario, trades, abiertas = h4.curva(velas, pd.DataFrame({t: [True, True, False, False] for t in velas}, index=fechas), senales)
    qr, qc = 250 / 10.3, (100_000 - 2500 * .0005) * .1 / 103
    entrada, salida = 103 * (qr + qc), 90 * (qr + qc)
    assert [t["cantidad"] for t in trades] == pytest.approx([qr, qc])
    assert diario.equity.iloc[2] == pytest.approx(100_000 + 9 * (qr + qc) - entrada * .0005)
    assert diario.equity.iloc[-1] == pytest.approx(100_000 + salida - entrada - (entrada + salida) * .0005)
    assert (diario.costos.sum(), diario.volumen.sum()) == pytest.approx(((entrada + salida) * .0005, entrada + salida))
    assert sum(t["pnl"] for t in trades) == pytest.approx(diario.equity.iloc[-1] - 100_000)
    assert diario.invertido.tolist() == [False, True, True, True] and not abiertas


def test_curva_heat_limite_rechazo_y_breakeven(monkeypatch):
    fechas, simbolos = pd.bdate_range("2011-01-03", periods=4), list("ABCDEFGHI")
    velas = {t: pd.DataFrame(dict(open=[100, 100, 100, 105], high=[100, 100, 105, 105],
                                low=[99, 99, 99, 104], close=[100, 100, 105, 105]), index=fechas) for t in simbolos}
    senales = {t: pd.DataFrame(dict(entry=[1, 0, 1, 0], entry_price=100, entry_expiry_bars=3,
                                  stop_loss_pct=.05, chandelier_dist=5), index=fechas) for t in simbolos}
    senales["I"]["stop_loss_pct"] = .024982498687417944  # heat justo bajo 2% antes del costo de entrada
    con_costos, _, _ = h4.curva(velas, pd.DataFrame(True, index=fechas, columns=simbolos), senales)
    assert con_costos.rechazos.iloc[1] == 2 and con_costos.heat.iloc[1] <= .02
    senales["I"]["stop_loss_pct"] = .05
    monkeypatch.setitem(h4.h3.COSTOS_PERFILES, "etf", dict(slippage_pct=0, commission_pct=0))
    diario, trades, abiertas = h4.curva(velas, pd.DataFrame(True, index=fechas, columns=simbolos), senales)
    assert diario.rechazos.tolist() == [0, 1, 0, 0] and len(abiertas) == 9 and not trades
    assert diario.heat.iloc[1] == pytest.approx(.02) and diario.heat.iloc[2] == 0
    assert abiertas["A"]["cantidad"] == 50 and abiertas["A"]["stop_loss"] == 100
    assert abiertas["I"]["cantidad"] == pytest.approx(255 / 5.25)


def test_curva_cancela_pendiente_al_salir_sin_vela():
    fechas = pd.bdate_range("2011-01-03", periods=3)
    velas = pd.DataFrame(dict(open=100, high=101, low=99, close=100), index=fechas[[0, 2]])
    senales = pd.DataFrame(dict(entry=[1, 0, 0], entry_price=100, entry_expiry_bars=3,
                               stop_loss_pct=.05, chandelier_dist=10), index=fechas)
    diario, trades, abiertas = h4.curva({"A": velas}, pd.DataFrame({"A": [True, False, True]}, index=fechas), {"A": senales})
    assert diario.equity.eq(100_000).all() and not diario.invertido.any() and not trades and not abiertas
