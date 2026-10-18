"""QuantAgent-d4cq.3: bar_signals convierte una estrategia en columnas de señales sin mirar el futuro."""

import numpy as np
import pandas as pd
import pytest

from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy
from quantagent.strategy.momentum_12m_strategy import Momentum12mStrategy
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy
from quantagent.strategy.sma_cross_strategy import SmaCrossStrategy


def _candles(closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=len(closes), freq="D"),
        "open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0,
    })


def test_sma_cross_entry_solo_en_el_cruce_y_exit_cuando_rapida_debajo():
    # fast=2, slow=3. Cruce arriba en la vela 5 (rápida 3.5 > lenta 3.0, antes 2.5 <= 3.0).
    closes = [5, 4, 3, 2, 2, 5, 5, 5, 1, 1]
    sig = bar_signals(SmaCrossStrategy(fast=2, slow=3), _candles(closes), "SPY", "1d")
    assert sig.index.equals(_candles(closes).index)
    assert list(sig["entry"]) == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # tras el cruce, la primera vela con rápida < lenta es la 8: (5+1)/2 = 3 contra (5+5+1)/3 = 3.67
    assert list(sig["exit"][5:]) == [False, False, False, True, True]
    assert sig["stop_loss_pct"].iloc[5] > 0 and sig["stop_loss_pct"].iloc[:5].isna().all()


@pytest.mark.parametrize("required", [4, 0, -1])
def test_la_estrategia_recibe_solo_la_ventana_requerida(required):
    class Espia(SmaCrossStrategy):
        def __init__(self):
            super().__init__(fast=2, slow=3)
            self.largos_signal, self.largos_exit = [], []

        @property
        def required_history_bars(self):
            return required

        def generate_signal(self, kline_data, symbol, timeframe, current_price):
            current_price = int(current_price)
            self.largos_signal.append(len(kline_data))
            assert [r["close"] for r in kline_data] == list(range(max(1, current_price - n + 1), current_price + 1))
            return None

        def should_exit(self, position, current_price, ohlc_data):
            current_price = int(current_price)
            self.largos_exit.append(len(ohlc_data))
            assert ohlc_data["close"].tolist() == list(range(max(1, current_price - n + 1), current_price + 1))
            return False, ""

    n = required if required > 0 else 30
    espia = Espia()
    bar_signals(espia, _candles(list(range(1, 36))), "SPY", "1d")
    assert espia.largos_signal == [min(i, n) for i in range(1, 36)]
    assert espia.largos_exit == [min(i, n) for i in range(1, 36)]


def test_rsi_produce_entradas_cortas_con_rsi_sobre_70():
    closes = [100 + i for i in range(30)]  # solo sube: RSI = 100
    sig = bar_signals(RSIMeanReversionStrategy(), _candles(closes), "SPY", "1d")
    assert (sig["entry"].iloc[:14] == 0).all()
    assert (sig["entry"].iloc[14:] == -1).all()
    assert sig["trailing_stop_pct"].iloc[14] == 0.05
    assert abs(sig["stop_loss_pct"].iloc[14] - 0.02) < 1e-9
    assert not sig["exit"].any()  # la regla por defecto con niveles inalcanzables nunca sale


def _serie_indexada(n=600):
    closes = 100 + np.cumsum(np.random.default_rng(7).normal(0, 1, n))
    c = _candles(closes)
    c["volume"] = np.random.default_rng(8).uniform(1, 3, n)
    return c.set_index("timestamp", drop=False)


def test_las_4_estrategias_de_m2_no_miran_el_futuro():
    candles = _serie_indexada()
    cuts = [candles.index[299], candles.index[449], candles.index[549]]
    estrategias = [
        RSIMeanReversionStrategy(), FiftyTwoWeekHighStrategy(lookback_days=60),
        SmaCrossStrategy(fast=10, slow=30), Momentum12mStrategy(lookback=60),
    ]
    for est in estrategias:
        assert check_no_lookahead(lambda c: bar_signals(est, c, "SPY", "1d"), candles, cuts) == [], type(est).__name__


def test_detecta_shift_negativo_y_normalizacion_con_toda_la_serie():
    candles = _serie_indexada(100)
    cuts = [candles.index[49], candles.index[79]]
    futuro = lambda c: pd.DataFrame({"s": (c["close"].shift(-1) > c["close"]).astype(int)})  # noqa: E731
    global_ = lambda c: pd.DataFrame({"z": (c["close"] - c["close"].mean()) / c["close"].std()})  # noqa: E731
    d1 = check_no_lookahead(futuro, candles, cuts)
    d2 = check_no_lookahead(global_, candles, cuts)
    assert d1 and d1[0].startswith("s:") and str(candles.index[49]) in d1[0]
    assert d2 and d2[0].startswith("z:")


def test_media_movil_sin_centrar_no_difiere():
    candles = _serie_indexada(100)
    cuts = [candles.index[49], candles.index[79]]
    causal = lambda c: pd.DataFrame({"m": c["close"].rolling(20).mean()})  # noqa: E731
    assert check_no_lookahead(causal, candles, cuts) == []
