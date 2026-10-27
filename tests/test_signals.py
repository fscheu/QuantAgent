"""QuantAgent-d4cq.3: bar_signals convierte una estrategia en columnas de señales sin mirar el futuro."""

import pandas as pd
import pytest

import numpy as np

from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.backtesting.vbt_filter import run_filter
from quantagent.strategy.base import TradingSignal, TradingStrategy
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


class _StopProbeStrategy(TradingStrategy):
    def __init__(self, entry_price: float = 105.0, expiry: int = 3):
        super().__init__()
        self.entry_price = entry_price
        self.expiry = expiry

    @property
    def required_history_bars(self) -> int:
        return 1

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        if kline_data[-1]["timestamp"] == pd.Timestamp("2020-01-01"):
            return TradingSignal(
                decision="LONG",
                confidence=1.0,
                entry_order="stop",
                entry_price=self.entry_price,
                entry_expiry_bars=self.expiry,
            )
        return None

    def should_reevaluate(self, position, current_price):
        return False


@pytest.mark.parametrize("fill_open, fill_high, expected_entry", [
    (102.0, 107.0, 105.0),
    (108.0, 110.0, 108.0),
])
def test_stop_entry_en_etapa_1(fill_open, fill_high, expected_entry):
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    strat = _StopProbeStrategy(entry_price=105.0, expiry=3)

    candles_no_cross = pd.DataFrame(
        {
            "timestamp": idx,
            "open": [100.0, 101.0, 102.0, 101.0],
            "high": [101.0, 104.0, 103.0, 102.0],
            "low": [99.0, 99.0, 100.0, 99.0],
            "close": [100.5, 102.0, 101.0, 100.0],
            "volume": 1.0,
        },
        index=idx,
    )
    sig_no_cross = bar_signals(strat, candles_no_cross, "TEST", "1d")
    assert sig_no_cross["entry_order"].iloc[0] == "stop"
    assert sig_no_cross["entry_price"].iloc[0] == 105.0
    assert sig_no_cross["entry_expiry_bars"].iloc[0] == 3.0
    assert (sig_no_cross["entry_order"].iloc[1:] == "market").all()
    assert sig_no_cross["entry_price"].iloc[1:].isna().all()
    assert sig_no_cross["entry_expiry_bars"].iloc[1:].isna().all()

    r_no_cross = run_filter(sig_no_cross, candles_no_cross, commission_pct=0.0, slippage_pct=0.0, capital=10_000.0)
    assert r_no_cross["trades"] == 0

    candles_cross = pd.DataFrame(
        {
            "timestamp": idx,
            "open": [100.0, fill_open, 106.0, 105.0],
            "high": [101.0, fill_high, 108.0, 107.0],
            "low": [99.0, 100.0, 104.0, 103.0],
            "close": [100.5, 106.0, 107.0, 106.0],
            "volume": 1.0,
        },
        index=idx,
    )
    sig_cross = bar_signals(strat, candles_cross, "TEST", "1d")
    r_cross = run_filter(sig_cross, candles_cross, commission_pct=0.0, slippage_pct=0.0, capital=10_000.0)
    assert r_cross["trades"] == 1
    assert float(r_cross["portfolio"].trades.records["entry_price"][0]) == pytest.approx(expected_entry)


def test_bar_signals_chandelier_dist():
    class DummyChandStrategy(SmaCrossStrategy):
        def __init__(self):
            super().__init__(fast=2, slow=3)
            self.chandelier_mult = 2.5
            self.atr_period = 3

    closes = [10.0, 11.0, 12.0, 13.0, 12.0]
    c = _candles(closes)
    sig_chand = bar_signals(DummyChandStrategy(), c, "TEST", "1d")
    assert "chandelier_dist" in sig_chand.columns
    assert sig_chand["chandelier_dist"].iloc[:2].isna().all()
    assert sig_chand["chandelier_dist"].iloc[2] == pytest.approx(2.5 * (2 / 3))

    sig_no_chand = bar_signals(SmaCrossStrategy(fast=2, slow=3), c, "TEST", "1d")
    assert "chandelier_dist" in sig_no_chand.columns
    assert sig_no_chand["chandelier_dist"].isna().all()

