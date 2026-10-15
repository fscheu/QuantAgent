"""QuantAgent-d4cq.3: bar_signals convierte una estrategia en columnas de señales sin mirar el futuro."""

import pandas as pd

from quantagent.backtesting.signals import bar_signals
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


def test_la_estrategia_recibe_en_la_vela_i_exactamente_i_mas_1_velas():
    class Espia(SmaCrossStrategy):
        def __init__(self):
            super().__init__(fast=2, slow=3)
            self.largos_signal, self.largos_exit = [], []

        def generate_signal(self, kline_data, symbol, timeframe, current_price):
            self.largos_signal.append(len(kline_data))
            return super().generate_signal(kline_data, symbol, timeframe, current_price)

        def should_exit(self, position, current_price, ohlc_data):
            self.largos_exit.append(len(ohlc_data))
            return super().should_exit(position, current_price, ohlc_data)

    espia = Espia()
    bar_signals(espia, _candles([1, 2, 3, 4, 5, 6, 7]), "SPY", "1d")
    assert espia.largos_signal == [1, 2, 3, 4, 5, 6, 7]
    assert espia.largos_exit == [1, 2, 3, 4, 5, 6, 7]


def test_rsi_produce_entradas_cortas_con_rsi_sobre_70():
    closes = [100 + i for i in range(30)]  # solo sube: RSI = 100
    sig = bar_signals(RSIMeanReversionStrategy(), _candles(closes), "SPY", "1d")
    assert (sig["entry"].iloc[:14] == 0).all()
    assert (sig["entry"].iloc[14:] == -1).all()
    assert sig["trailing_stop_pct"].iloc[14] == 0.05
    assert abs(sig["stop_loss_pct"].iloc[14] - 0.02) < 1e-9
    assert not sig["exit"].any()  # la regla por defecto con niveles inalcanzables nunca sale
