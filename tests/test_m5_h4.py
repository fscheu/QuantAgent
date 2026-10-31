import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
