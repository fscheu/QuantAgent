"""Tests for TripleScreenStrategy — QuantAgent-vna."""

import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
from unittest.mock import Mock, patch

import pandas as pd

from quantagent.backtesting.backtest import Backtest
from quantagent.models import ActivePosition, ExitPolicy, MarketData, OrderSide
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(
    prices: List[float],
    spread: float = 0.5,
) -> List[Dict]:
    """Build OHLCV candle list from a close-price sequence."""
    candles = []
    for i, p in enumerate(prices):
        candles.append(
            {
                "timestamp": datetime(2024, 1, 1) + timedelta(hours=i * 4),
                "open": p - spread * 0.5,
                "high": p + spread,
                "low": p - spread,
                "close": p,
                "volume": 1000.0,
            }
        )
    return candles


def _uptrend_candles(n: int = 200, base: float = 100.0) -> List[Dict]:
    """
    Produces n candles in a durable uptrend with a mild pullback at the end.

    The pullback is shallow enough to keep Screen 1 trending UP after weekly
    aggregation, while still pushing the fast stochastic below the oversold
    threshold for Screen 2.
    """
    trend_prices = [base + i * 0.5 for i in range(n - 20)]
    pullback_prices = [trend_prices[-1] - i * 0.5 for i in range(1, 21)]
    return _make_candles(trend_prices + pullback_prices)


def _downtrend_candles(n: int = 200, base: float = 200.0) -> List[Dict]:
    """
    Produces n candles in a durable downtrend with a mild rally at the end.

    The rally is shallow enough to keep Screen 1 trending DOWN after weekly
    aggregation, while still pushing the fast stochastic above the overbought
    threshold for Screen 2.
    """
    trend_prices = [base - i * 0.5 for i in range(n - 20)]
    rally_prices = [trend_prices[-1] + i * 0.5 for i in range(1, 21)]
    return _make_candles(trend_prices + rally_prices)


def _seed_market_data(db_session, symbol: str, timeframe: str, candles: List[Dict]) -> None:
    for candle in candles:
        db_session.add(
            MarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=candle["timestamp"],
                open=Decimal(str(candle["open"])),
                high=Decimal(str(candle["high"])),
                low=Decimal(str(candle["low"])),
                close=Decimal(str(candle["close"])),
                volume=Decimal(str(candle["volume"])),
            )
        )
    db_session.commit()


# ---------------------------------------------------------------------------
# AC1 — Class validity
# ---------------------------------------------------------------------------

class TestClassIsValidStrategy:
    """AC1: TripleScreenStrategy satisfies the TradingStrategy ABC."""

    def test_class_is_valid_strategy(self):
        assert issubclass(TripleScreenStrategy, TradingStrategy)

    def test_instantiation_with_defaults(self):
        strategy = TripleScreenStrategy()
        assert strategy is not None

    def test_instantiation_with_custom_params(self):
        strategy = TripleScreenStrategy(
            weekly_bars=3,
            trend_ema_period=10,
            stoch_k_period=4,
            stoch_d_period=2,
            stoch_oversold=25.0,
            stoch_overbought=75.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            trailing_stop_pct=0.07,
        )
        assert strategy.weekly_bars == 3
        assert strategy.stoch_oversold == 25.0


# ---------------------------------------------------------------------------
# AC2 — Screen 1: EMA slope
# ---------------------------------------------------------------------------

class TestScreen1Trend:
    """AC2: Screen 1 correctly classifies trend direction."""

    def _weekly_df_from_prices(self, closes: List[float]) -> pd.DataFrame:
        rows = [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000.0} for c in closes]
        return pd.DataFrame(rows)

    def test_screen1_uptrend(self):
        """AC2.1: Rising closes → UP."""
        strategy = TripleScreenStrategy(trend_ema_period=5)
        closes = [100.0 + i for i in range(20)]
        weekly_df = self._weekly_df_from_prices(closes)
        assert strategy._screen1_trend(weekly_df) == "UP"

    def test_screen1_downtrend(self):
        """AC2.2: Falling closes → DOWN."""
        strategy = TripleScreenStrategy(trend_ema_period=5)
        closes = [120.0 - i for i in range(20)]
        weekly_df = self._weekly_df_from_prices(closes)
        assert strategy._screen1_trend(weekly_df) == "DOWN"

    def test_screen1_insufficient_bars_returns_none(self):
        """AC2.3: Fewer than trend_ema_period + 1 bars → None."""
        strategy = TripleScreenStrategy(trend_ema_period=13)
        closes = [100.0] * 5  # only 5 bars, need 14
        weekly_df = self._weekly_df_from_prices(closes)
        assert strategy._screen1_trend(weekly_df) is None


# ---------------------------------------------------------------------------
# AC3 — Screen 2: Stochastic
# ---------------------------------------------------------------------------

class TestScreen2Oscillator:
    """AC3: Screen 2 stochastic activation rules."""

    def _oversold_candles(self, n: int = 30) -> List[Dict]:
        """Last few candles at very low prices → stoch_k << 20."""
        high_prices = [100.0] * (n - 8)
        low_prices = [60.0 - i for i in range(8)]
        return _make_candles(high_prices + low_prices)

    def _overbought_candles(self, n: int = 30) -> List[Dict]:
        """Last few candles at very high prices → stoch_k >> 80."""
        low_prices = [60.0] * (n - 8)
        high_prices = [100.0 + i for i in range(8)]
        return _make_candles(low_prices + high_prices)

    def _midrange_candles(self, n: int = 30, price: float = 80.0) -> List[Dict]:
        """Candles in mid-range → stoch_k ~50."""
        prices = [price + (i % 3) * 0.1 for i in range(n)]
        return _make_candles(prices)

    def test_screen2_uptrend_oversold_activates(self):
        """AC3.1: Uptrend + oversold stoch → True."""
        strategy = TripleScreenStrategy(stoch_k_period=5, stoch_d_period=3)
        df = pd.DataFrame(self._oversold_candles())
        assert strategy._screen2_oscillator(df, "UP") is True

    def test_screen2_uptrend_not_oversold(self):
        """AC3.2: Uptrend + mid-range stoch → False."""
        strategy = TripleScreenStrategy(stoch_k_period=5, stoch_d_period=3)
        df = pd.DataFrame(self._midrange_candles())
        assert strategy._screen2_oscillator(df, "UP") is False

    def test_screen2_downtrend_overbought_activates(self):
        """AC3.3: Downtrend + overbought stoch → True."""
        strategy = TripleScreenStrategy(stoch_k_period=5, stoch_d_period=3)
        df = pd.DataFrame(self._overbought_candles())
        assert strategy._screen2_oscillator(df, "DOWN") is True

    def test_screen2_downtrend_not_overbought(self):
        """AC3.4: Downtrend + mid-range stoch → False."""
        strategy = TripleScreenStrategy(stoch_k_period=5, stoch_d_period=3)
        df = pd.DataFrame(self._midrange_candles())
        assert strategy._screen2_oscillator(df, "DOWN") is False


# ---------------------------------------------------------------------------
# AC4 — Screen 3: Breakout trigger
# ---------------------------------------------------------------------------

class TestScreen3Trigger:
    """AC4: Screen 3 breakout fires correctly."""

    def _candles_with_prior_high(self, prior_high: float) -> List[Dict]:
        candles = _make_candles([100.0] * 10)
        candles[-2]["high"] = prior_high
        candles[-1]["close"] = prior_high - 2
        return candles

    def _candles_with_prior_low(self, prior_low: float) -> List[Dict]:
        candles = _make_candles([100.0] * 10)
        candles[-2]["low"] = prior_low
        candles[-1]["close"] = prior_low + 2
        return candles

    def test_screen3_long_breakout_triggers(self):
        """AC4.1: current_price > prior high → True (UP)."""
        strategy = TripleScreenStrategy()
        klines = self._candles_with_prior_high(prior_high=105.0)
        assert strategy._screen3_trigger(klines, "UP", 106.0) is True

    def test_screen3_long_no_breakout(self):
        """AC4.2: current_price <= prior high → False (UP)."""
        strategy = TripleScreenStrategy()
        klines = self._candles_with_prior_high(prior_high=105.0)
        assert strategy._screen3_trigger(klines, "UP", 104.9) is False

    def test_screen3_short_breakout_triggers(self):
        """AC4.3: current_price < prior low → True (DOWN)."""
        strategy = TripleScreenStrategy()
        klines = self._candles_with_prior_low(prior_low=95.0)
        assert strategy._screen3_trigger(klines, "DOWN", 94.9) is True

    def test_screen3_short_no_breakout(self):
        """AC4.4: current_price >= prior low → False (DOWN)."""
        strategy = TripleScreenStrategy()
        klines = self._candles_with_prior_low(prior_low=95.0)
        assert strategy._screen3_trigger(klines, "DOWN", 95.1) is False


# ---------------------------------------------------------------------------
# AC5 — generate_signal integration
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    """AC5: generate_signal combines all three screens correctly."""

    def test_generate_signal_long_all_screens_pass(self):
        """AC5.1: All screens → LONG signal with correct fields."""
        strategy = TripleScreenStrategy()
        klines = _uptrend_candles(n=200)

        # current_price must be above kline[-2]["high"] to fire Screen 3
        prior_high = float(klines[-2]["high"])
        current_price = prior_high + 1.0

        signal = strategy.generate_signal(klines, "BTC-USD", "4h", current_price)

        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.decision == "LONG"
        assert 0.1 <= signal.confidence <= 1.0
        assert signal.entry_price == current_price
        assert abs(signal.stop_loss - current_price * (1 - strategy.stop_loss_pct)) < 1e-6
        assert abs(signal.take_profit - current_price * (1 + strategy.take_profit_pct)) < 1e-6
        assert "Screen" in signal.reasoning

    def test_generate_signal_short_all_screens_pass(self):
        """AC5.2: All screens → SHORT signal with correct fields."""
        strategy = TripleScreenStrategy()
        klines = _downtrend_candles(n=200)

        prior_low = float(klines[-2]["low"])
        current_price = prior_low - 1.0

        signal = strategy.generate_signal(klines, "BTC-USD", "4h", current_price)

        assert signal is not None
        assert signal.decision == "SHORT"
        assert abs(signal.stop_loss - current_price * (1 + strategy.stop_loss_pct)) < 1e-6
        assert abs(signal.take_profit - current_price * (1 - strategy.take_profit_pct)) < 1e-6

    def test_generate_signal_screen2_fails_returns_none(self):
        """AC5.3: Screen 1 UP but stoch not oversold → None."""
        strategy = TripleScreenStrategy()
        # Steady uptrend with no pullback: stoch will be near 100 (overbought), not oversold
        prices = [100.0 + i for i in range(200)]
        klines = _make_candles(prices)
        current_price = float(klines[-2]["high"]) + 1.0
        signal = strategy.generate_signal(klines, "BTC-USD", "4h", current_price)
        assert signal is None

    def test_generate_signal_screen3_fails_returns_none(self):
        """AC5.4: Screens 1+2 pass but no breakout → None."""
        strategy = TripleScreenStrategy()
        klines = _uptrend_candles(n=200)

        # current_price BELOW prior high → Screen 3 fails
        prior_high = float(klines[-2]["high"])
        current_price = prior_high - 1.0

        signal = strategy.generate_signal(klines, "BTC-USD", "4h", current_price)
        assert signal is None

    def test_generate_signal_insufficient_candles_returns_none(self):
        """AC5.5: Fewer than min_candles → None without exception."""
        strategy = TripleScreenStrategy()
        klines = _make_candles([100.0] * 10)
        signal = strategy.generate_signal(klines, "BTC-USD", "4h", 100.0)
        assert signal is None

    def test_generate_signal_exactly_min_candles(self):
        """Edge: exactly min_candles does not raise."""
        strategy = TripleScreenStrategy()
        n = strategy._min_candles
        klines = _make_candles([100.0] * n)
        # Should not raise — may return None depending on market conditions
        result = strategy.generate_signal(klines, "BTC-USD", "4h", 100.5)
        assert result is None or isinstance(result, TradingSignal)


# ---------------------------------------------------------------------------
# AC6 — Realistic scenario generates non-HOLD signal
# ---------------------------------------------------------------------------

def test_generates_signal_in_realistic_scenario():
    """AC6: Strategy generates a real signal (not None) in designed scenario."""
    strategy = TripleScreenStrategy()
    klines = _uptrend_candles(n=200)
    prior_high = float(klines[-2]["high"])
    current_price = prior_high + 1.0
    signal = strategy.generate_signal(klines, "BTC-USD", "4h", current_price)
    assert signal is not None
    assert signal.decision in ("LONG", "SHORT")


# ---------------------------------------------------------------------------
# AC8 — should_reevaluate always False
# ---------------------------------------------------------------------------

def test_should_reevaluate_false():
    """AC8: should_reevaluate returns False unconditionally."""
    strategy = TripleScreenStrategy()
    position = ActivePosition(
        id=1,
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        quantity=1.0,
        decision_timestamp=datetime.utcnow(),
        candles_since_entry=5,
        exit_policy=ExitPolicy.TRAILING_STOP,
        prediction_horizon=3,
        candles_direction=[],
        is_active=True,
    )
    assert strategy.should_reevaluate(position, 102.0) is False


def test_backtest_reference_profile_completes(db_session):
    """AC7: Reference backtest profile completes and produces finite metrics."""
    strategy = TripleScreenStrategy()
    seed_start = datetime(2023, 12, 1)
    prices = [100.0 + i * 0.15 for i in range(732)]
    candles = _make_candles(prices)
    for index, candle in enumerate(candles):
        candle["timestamp"] = seed_start + timedelta(hours=index * 4)

    _seed_market_data(db_session, "BTC-USD", "4h", candles)

    config = {
        "base_position_pct": 0.05,
        "max_daily_loss_pct": 0.05,
        "max_position_pct": 0.10,
        "slippage_pct": 0.01,
        "agent_llm_provider": "openai",
        "agent_llm_model": "gpt-4o-mini",
        "agent_llm_temperature": 0.1,
        "market_hours_filter": False,
    }

    with patch("quantagent.data.provider.yf.download") as mock_download:
        mock_download.return_value = Mock(empty=True)
        backtest = Backtest(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            assets=["BTC-USD"],
            timeframe="4h",
            initial_capital=100_000.0,
            config=config,
            db_session=db_session,
            strategy=strategy,
        )
        metrics = backtest.run(name="QuantAgent-vna-reference")

    assert math.isfinite(metrics.total_pnl)
    assert metrics.total_trades >= 0


# ---------------------------------------------------------------------------
# Exit policy
# ---------------------------------------------------------------------------

def test_default_exit_policy():
    strategy = TripleScreenStrategy()
    assert strategy.get_default_exit_policy() == ExitPolicy.TRAILING_STOP
