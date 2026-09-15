from datetime import datetime, timedelta

import pandas as pd

from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy
from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy


def _build_triple_screen_fixture():
    current = datetime(2025, 1, 1)
    rows = []
    close = 100.0

    # Long uptrend, then pullback hard enough to flip trend in the synthetic weekly aggregation.
    for _ in range(120):
        rows.append(
            {
                "timestamp": current,
                "open": close - 0.8,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1500,
            }
        )
        current += timedelta(hours=4)
        close += 1.0

    for _ in range(5):
        rows.append(
            {
                "timestamp": current,
                "open": close - 0.5,
                "high": close + 0.3,
                "low": close - 10.0,
                "close": close - 8.0,
                "volume": 1500,
            }
        )
        current += timedelta(hours=4)
        close -= 8.0

    prior_close = close + 1.0
    rows.append(
        {
            "timestamp": current,
            "open": prior_close - 0.5,
            "high": prior_close + 0.5,
            "low": prior_close - 1.0,
            "close": prior_close,
            "volume": 1500,
        }
    )
    current += timedelta(hours=4)

    breakout_close = prior_close + 2.0
    rows.append(
        {
            "timestamp": current,
            "open": breakout_close - 0.5,
            "high": breakout_close + 0.5,
            "low": breakout_close - 0.8,
            "close": breakout_close,
            "volume": 1600,
        }
    )
    return rows, breakout_close


def _build_52w_fixture():
    current = datetime(2024, 1, 1)
    rows = []
    idx = 0

    while len(rows) < 320:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        if idx < 300:
            close = 100 + (idx * 0.08)
            high = close + 0.8
            volume = 1000
        elif idx < 319:
            close = 123.0 + ((idx - 300) * 0.05)
            high = close + 0.6
            volume = 1000
        else:
            close = 126.5
            high = 127.0
            volume = 2500

        rows.append(
            {
                "timestamp": current,
                "open": close - 0.4,
                "high": high,
                "low": close - 1.0,
                "close": close,
                "volume": volume,
            }
        )
        idx += 1
        current += timedelta(days=1)

    return rows, rows[-1]["close"]


def test_triple_screen_fixture_conditions_explain_why_it_does_not_trade_yet():
    strategy = TripleScreenStrategy()
    kline_data, current_price = _build_triple_screen_fixture()

    df = pd.DataFrame(kline_data)
    weekly_df = strategy._aggregate_weekly_bars(df)
    trend = strategy._screen1_trend(weekly_df)
    screen2 = strategy._screen2_oscillator(df, trend) if trend else None
    screen3 = strategy._screen3_trigger(kline_data, trend, current_price) if trend else None
    signal = strategy.generate_signal(kline_data, "BTC", "4h", current_price)

    assert trend == "DOWN"
    assert screen2 is False
    assert screen3 is False
    assert signal is None


def test_fifty_two_week_fixture_emits_long_signal():
    strategy = FiftyTwoWeekHighStrategy()
    kline_data, current_price = _build_52w_fixture()

    signal = strategy.generate_signal(kline_data, "SPX", "1d", current_price)

    assert signal is not None
    assert signal.decision == "LONG"
