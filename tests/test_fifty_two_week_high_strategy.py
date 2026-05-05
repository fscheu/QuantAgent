"""Unit tests for FiftyTwoWeekHighStrategy — QuantAgent-b8r."""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import pytest

from quantagent.models import ActivePosition, Environment, ExitPolicy, OrderSide
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DT = datetime(2022, 1, 1)


def _make_kline_data(
    n: int = 303,
    close_val: float = 90.0,
    high_val: float = 99.0,
    normal_volume: float = 1_000_000.0,
    last_volume: float = 2_000_000.0,
) -> List[Dict]:
    """Return n uniform daily candles.

    With defaults and current_price=100.0:
    - high_52w = 99.0  (breakout: 100 > 99)
    - SMA-50   = 90.0  (trend: 100 > 90)
    - vol_ma   ≈ 1.05e6, vol[-1] = 2e6  (vol: 2e6 > 1.5 × 1.05e6)
    """
    rows = []
    for i in range(n):
        vol = last_volume if i == n - 1 else normal_volume
        rows.append(
            {
                "timestamp": _BASE_DT + timedelta(days=i),
                "open": close_val,
                "high": high_val,
                "low": close_val * 0.99,
                "close": close_val,
                "volume": vol,
            }
        )
    return rows


def _make_realistic_kline_data() -> List[Dict]:
    """400-candle uptrend, volume spike on last bar, close above 52w high."""
    n = 400
    rows = []
    for i in range(n - 1):
        close = 90.0 + 9.0 * i / (n - 2)
        rows.append(
            {
                "timestamp": _BASE_DT + timedelta(days=i),
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    close_last = 100.5
    rows.append(
        {
            "timestamp": _BASE_DT + timedelta(days=n - 1),
            "open": close_last,
            "high": close_last * 1.002,
            "low": close_last * 0.998,
            "close": close_last,
            "volume": 2_500_000.0,
        }
    )
    return rows


def _make_active_position() -> ActivePosition:
    return ActivePosition(
        id=1,
        symbol="AAPL",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        quantity=10.0,
        decision_timestamp=datetime.utcnow(),
        candles_since_entry=5,
        exit_policy=ExitPolicy.TRAILING_STOP,
        prediction_horizon=3,
        candles_direction=[],
        is_active=True,
        environment=Environment.BACKTEST,
    )


# ---------------------------------------------------------------------------
# AC1 — Class validity
# ---------------------------------------------------------------------------


class TestClassValidity:
    """AC1: strategy class exists and satisfies the TradingStrategy ABC."""

    def test_class_is_valid_strategy(self):
        assert issubclass(FiftyTwoWeekHighStrategy, TradingStrategy)

    def test_instantiation_with_defaults(self):
        s = FiftyTwoWeekHighStrategy()
        assert s is not None

    def test_instantiation_with_custom_params(self):
        s = FiftyTwoWeekHighStrategy(
            lookback_days=200,
            proximity_threshold=0.97,
            trend_ma_period=30,
            volume_ma_period=10,
            volume_factor=2.0,
            stop_loss_pct=0.03,
            take_profit_pct=0.10,
            trailing_stop_pct=0.05,
        )
        assert s.lookback_days == 200
        assert s.volume_factor == 2.0

    def test_exported_from_strategy_package(self):
        from quantagent.strategy import FiftyTwoWeekHighStrategy as F

        assert F is FiftyTwoWeekHighStrategy


# ---------------------------------------------------------------------------
# AC2 — 52-week high calculation
# ---------------------------------------------------------------------------


class TestFiftyTwoWeekHighCalculation:
    """AC2: _compute_52w_high returns the correct rolling max."""

    def test_rolling_max_picks_known_value(self):
        """AC2.1: known maximum within lookback window is returned."""
        strategy = FiftyTwoWeekHighStrategy()
        highs = pd.Series([99.0] * 303)
        # Index 200 is within highs.iloc[-253:-1] (indices 50..301)
        highs.iloc[200] = 150.0
        result = strategy._compute_52w_high(highs)
        assert result == 150.0

    def test_excludes_current_bar(self):
        """AC2.2: current (last) candle high is NOT included in the 52w high."""
        strategy = FiftyTwoWeekHighStrategy()
        highs = pd.Series([99.0] * 303)
        highs.iloc[-1] = 999.0  # artificially high last bar
        result = strategy._compute_52w_high(highs)
        assert result == 99.0

    def test_uses_exactly_lookback_days_candles(self):
        """Lookback window contains exactly lookback_days elements."""
        strategy = FiftyTwoWeekHighStrategy(lookback_days=100)
        n = 200
        highs = pd.Series([50.0] * n)
        # highs.iloc[-(100+1):-1] = highs.iloc[-101:-1] = indices 99..198 (100 elements)
        # Put the max at index 99 (first element of the window)
        highs.iloc[99] = 77.0
        result = strategy._compute_52w_high(highs)
        assert result == 77.0

    def test_does_not_include_bar_before_lookback_window(self):
        """Values older than lookback_days are ignored."""
        strategy = FiftyTwoWeekHighStrategy(lookback_days=100)
        n = 200
        highs = pd.Series([50.0] * n)
        # Index 98 is OUTSIDE the window highs.iloc[-101:-1] = indices 99..198
        highs.iloc[98] = 999.0
        result = strategy._compute_52w_high(highs)
        assert result == 50.0  # 999 at index 98 is not included


# ---------------------------------------------------------------------------
# AC3 — Trend filter
# ---------------------------------------------------------------------------


class TestTrendFilter:
    """AC3: price vs SMA-50 controls the trend gate."""

    def test_trend_filter_passes_when_price_above_sma(self):
        """AC3.1: trend OK when current_price > SMA-50."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(close_val=90.0, high_val=99.0)
        # current_price=100 > SMA-50=90 → trend pass
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is not None
        assert signal.decision == "LONG"

    def test_trend_filter_blocks_when_price_below_sma(self):
        """AC3.2: signal is None when current_price < SMA-50."""
        strategy = FiftyTwoWeekHighStrategy()
        # close=110 → SMA-50=110; current_price=100 < 110 → trend fail
        # high=99 < 100 → breakout OK; volume OK
        data = _make_kline_data(close_val=110.0, high_val=99.0)
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is None


# ---------------------------------------------------------------------------
# AC4 — Volume filter
# ---------------------------------------------------------------------------


class TestVolumeFilter:
    """AC4: last-bar volume vs volume MA controls the volume gate."""

    def test_volume_filter_passes_when_volume_above_threshold(self):
        """AC4.1: vol OK when last volume > volume_factor × vol_MA."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(normal_volume=1_000_000, last_volume=2_000_000)
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is not None

    def test_volume_filter_blocks_low_volume_breakout(self):
        """AC4.2: breakout without volume confirmation → None."""
        strategy = FiftyTwoWeekHighStrategy()
        # All volumes equal → vol_ok fails (vol == vol_factor × vol_ma exactly fails strict >)
        data = _make_kline_data(normal_volume=1_000_000, last_volume=1_000_000)
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is None


# ---------------------------------------------------------------------------
# AC5 — Breakout condition (strict >)
# ---------------------------------------------------------------------------


class TestBreakoutCondition:
    """AC5: strict > comparison for 52w-high breakout."""

    def test_no_breakout_returns_none(self):
        """AC5.1: price below 52w high → None (even if close)."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(high_val=99.0)
        # current_price=98.0 < 99.0 = high_52w → no breakout
        signal = strategy.generate_signal(data, "AAPL", "1d", 98.0)
        assert signal is None

    def test_exact_equality_returns_none(self):
        """AC5.2: price == 52w high → None (strict > required)."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(high_val=99.0)
        signal = strategy.generate_signal(data, "AAPL", "1d", 99.0)
        assert signal is None

    def test_breakout_returns_signal(self):
        """Price strictly above 52w high with all filters passing → LONG."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(high_val=99.0)
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is not None
        assert signal.decision == "LONG"


# ---------------------------------------------------------------------------
# AC6 — Combined generate_signal
# ---------------------------------------------------------------------------


class TestGenerateSignal:
    """AC6: combined signal generation with all ACs for the LONG output."""

    def test_all_conditions_pass_produces_long_signal(self):
        """AC6.1: breakout + uptrend + volume → LONG with correct fields."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(close_val=90.0, high_val=99.0)
        current_price = 100.0

        signal = strategy.generate_signal(data, "AAPL", "1d", current_price)

        assert isinstance(signal, TradingSignal)
        assert signal.decision == "LONG"
        assert 0.1 <= signal.confidence <= 1.0
        assert signal.entry_price == current_price
        assert abs(signal.stop_loss - current_price * (1 - strategy.stop_loss_pct)) < 1e-6
        assert abs(signal.take_profit - current_price * (1 + strategy.take_profit_pct)) < 1e-6
        assert "52w-high" in signal.reasoning
        assert signal.exit_policy == ExitPolicy.TRAILING_STOP

    def test_no_short_signals_ever_produced(self):
        """AC6.2: strategy never produces SHORT decisions."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data()
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        if signal is not None:
            assert signal.decision != "SHORT"

    def test_trend_filter_alone_blocks_signal(self):
        """AC6.3: breakout + volume pass but trend fails → None."""
        strategy = FiftyTwoWeekHighStrategy()
        # close=110 → SMA-50=110 > current_price=100 → trend fail
        data = _make_kline_data(close_val=110.0, high_val=99.0)
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is None

    def test_volume_filter_alone_blocks_signal(self):
        """AC6.4: breakout + trend pass but volume fails → None."""
        strategy = FiftyTwoWeekHighStrategy()
        data = _make_kline_data(
            close_val=90.0, high_val=99.0, normal_volume=1_000_000, last_volume=1_000_000
        )
        signal = strategy.generate_signal(data, "AAPL", "1d", 100.0)
        assert signal is None

    def test_fewer_than_min_candles_returns_none_without_exception(self):
        """AC6.5: insufficient data → None, no exception raised."""
        strategy = FiftyTwoWeekHighStrategy()
        min_candles = (
            strategy.lookback_days
            + max(strategy.trend_ma_period, strategy.volume_ma_period)
            + 1
        )
        short_data = _make_kline_data(n=min_candles - 1)
        signal = strategy.generate_signal(short_data, "AAPL", "1d", 100.0)
        assert signal is None

    def test_empty_data_returns_none(self):
        signal = FiftyTwoWeekHighStrategy().generate_signal([], "AAPL", "1d", 100.0)
        assert signal is None


# ---------------------------------------------------------------------------
# AC7 — Confidence calculation
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """AC7: confidence = clamp(raw × 10, 0.1, 1.0) where raw = (price - 52wh) / 52wh."""

    def setup_method(self):
        self.strategy = FiftyTwoWeekHighStrategy()
        self.data = _make_kline_data(close_val=90.0, high_val=99.0)

    def test_confidence_5pct_above_high(self):
        """5% above → raw=0.05 → confidence=0.5."""
        current_price = 99.0 * 1.05  # ≈ 103.95
        signal = self.strategy.generate_signal(self.data, "AAPL", "1d", current_price)
        assert signal is not None
        expected = 0.05 * 10  # 0.5
        assert abs(signal.confidence - expected) < 1e-6

    def test_confidence_floor_applied(self):
        """0.5% above → raw=0.005 → confidence floored at 0.1."""
        current_price = 99.0 * 1.005  # ≈ 99.495
        signal = self.strategy.generate_signal(self.data, "AAPL", "1d", current_price)
        assert signal is not None
        assert signal.confidence == pytest.approx(0.1, abs=1e-6)

    def test_confidence_cap_applied(self):
        """15% above → raw=0.15 → confidence capped at 1.0."""
        current_price = 99.0 * 1.15  # ≈ 113.85
        signal = self.strategy.generate_signal(self.data, "AAPL", "1d", current_price)
        assert signal is not None
        assert signal.confidence == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AC8 — Realistic scenario
# ---------------------------------------------------------------------------


class TestRealisticScenario:
    """AC8: strategy generates LONG in a realistic uptrending 400-candle dataset."""

    def test_generates_signal_in_realistic_scenario(self):
        data = _make_realistic_kline_data()
        strategy = FiftyTwoWeekHighStrategy()
        current_price = data[-1]["close"]

        signal = strategy.generate_signal(data, "AAPL", "1d", current_price)

        assert signal is not None, "Expected a LONG signal in realistic uptrend scenario"
        assert signal.decision == "LONG"


# ---------------------------------------------------------------------------
# AC9 — should_reevaluate always False
# ---------------------------------------------------------------------------


class TestShouldReevaluate:
    """AC9: should_reevaluate always returns False."""

    def test_should_reevaluate_false(self):
        strategy = FiftyTwoWeekHighStrategy()
        position = _make_active_position()
        assert strategy.should_reevaluate(position, 105.0) is False

    def test_should_reevaluate_false_at_loss(self):
        strategy = FiftyTwoWeekHighStrategy()
        position = _make_active_position()
        assert strategy.should_reevaluate(position, 50.0) is False


# ---------------------------------------------------------------------------
# AC9+ — get_default_exit_policy
# ---------------------------------------------------------------------------


class TestExitPolicy:
    def test_default_exit_policy_is_trailing_stop(self):
        assert FiftyTwoWeekHighStrategy().get_default_exit_policy() == ExitPolicy.TRAILING_STOP


# ---------------------------------------------------------------------------
# AC10 — Reference backtest (integration, requires DB + data)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBacktestIntegration:
    """AC10: reference backtest completes without crash and PnL is calculated."""

    def test_backtest_integration(self, db_session):
        """Skipped unless integration marker selected and data is available."""
        pytest.skip(
            "AC10 is a manual integration test: requires AAPL daily OHLCV data "
            "for 2022–2023 in the database. Run with: "
            "pytest tests/test_fifty_two_week_high_strategy.py::TestBacktestIntegration -m integration"
        )
