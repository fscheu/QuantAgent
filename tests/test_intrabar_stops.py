"""Unit tests for intrabar stop loss and take profit evaluation (QuantAgent-al2.1)."""

from datetime import datetime
from decimal import Decimal
import pytest

from quantagent.backtesting.backtest import check_intrabar_stops, _is_candle_after_entry
from quantagent.models import OrderSide


@pytest.mark.parametrize(
    "side,sl,tp,candle,expected",
    [
        # Long scenarios
        (OrderSide.BUY, 98.0, 105.0, {"open": 100.0, "high": 102.0, "low": 97.5, "close": 99.0}, ("STOP_LOSS", 98.0)),
        (OrderSide.BUY, 98.0, 105.0, {"open": 100.0, "high": 105.5, "low": 99.0, "close": 104.0}, ("TAKE_PROFIT", 105.0)),
        ("buy", 98.0, 105.0, {"open": 100.0, "high": 106.0, "low": 97.0, "close": 101.0}, ("STOP_LOSS", 98.0)),
        (OrderSide.BUY, 98.0, 105.0, {"open": 96.0, "high": 99.0, "low": 95.0, "close": 98.5}, ("STOP_LOSS", 96.0)),
        ("long", 98.0, 105.0, {"open": 106.5, "high": 108.0, "low": 105.0, "close": 107.0}, ("TAKE_PROFIT", 106.5)),
        (OrderSide.BUY, 98.0, 105.0, {"open": 100.0, "high": 103.0, "low": 99.0, "close": 101.0}, None),
        # Short scenarios
        (OrderSide.SELL, 102.0, 95.0, {"open": 100.0, "high": 102.5, "low": 98.0, "close": 101.0}, ("STOP_LOSS", 102.0)),
        (OrderSide.SELL, 102.0, 95.0, {"open": 100.0, "high": 101.0, "low": 94.5, "close": 96.0}, ("TAKE_PROFIT", 95.0)),
        ("sell", 102.0, 95.0, {"open": 100.0, "high": 103.0, "low": 94.0, "close": 99.0}, ("STOP_LOSS", 102.0)),
        (OrderSide.SELL, 102.0, 95.0, {"open": 103.5, "high": 104.0, "low": 101.0, "close": 102.5}, ("STOP_LOSS", 103.5)),
        ("short", 102.0, 95.0, {"open": 93.5, "high": 96.0, "low": 93.0, "close": 94.5}, ("TAKE_PROFIT", 93.5)),
        (OrderSide.SELL, 102.0, 95.0, {"open": 100.0, "high": 101.0, "low": 98.0, "close": 99.0}, None),
    ],
    ids=[
        "long_sl_hit", "long_tp_hit", "long_both_hit", "long_gap_sl", "long_gap_tp", "long_no_hit",
        "short_sl_hit", "short_tp_hit", "short_both_hit", "short_gap_sl", "short_gap_tp", "short_no_hit",
    ],
)
def test_check_intrabar_stops_scenarios(side, sl, tp, candle, expected):
    pos = {"side": side, "stop_loss": sl, "take_profit": tp}
    assert check_intrabar_stops(pos, candle) == expected


def test_check_intrabar_stops_with_object_and_decimals():
    class PositionStub:
        side = OrderSide.BUY
        stop_loss = Decimal("450.50000000")
        take_profit = Decimal("550.00000000")

    class CandleStub:
        open = 500.0
        high = 520.0
        low = 450.0
        close = 480.0

    assert check_intrabar_stops(PositionStub(), CandleStub()) == ("STOP_LOSS", 450.5)


def test_is_candle_after_entry():
    entry_time = datetime(2026, 1, 2, 5, 0)
    assert not _is_candle_after_entry(datetime(2026, 1, 2, 5, 0), entry_time)
    assert not _is_candle_after_entry(datetime(2026, 1, 2, 4, 0), entry_time)
    assert _is_candle_after_entry(datetime(2026, 1, 2, 6, 0), entry_time)
    assert _is_candle_after_entry(datetime(2026, 1, 2, 6, 0), None)


def test_backtest_engine_intrabar_stops_defaults():
    from unittest.mock import MagicMock
    from quantagent.backtesting.backtest import Backtest

    # Default should be True
    bt_default = Backtest(
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 1, 2),
        assets=["SPY"],
        timeframe="1h",
        db_session=MagicMock(),
    )
    assert bt_default.intrabar_stops is True

    # Explicit False should be False
    bt_disabled = Backtest(
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 1, 2),
        assets=["SPY"],
        timeframe="1h",
        db_session=MagicMock(),
        intrabar_stops=False,
    )
    assert bt_disabled.intrabar_stops is False

    # Config False should be False
    bt_config = Backtest(
        start_date=datetime(2026, 1, 1),
        end_date=datetime(2026, 1, 2),
        assets=["SPY"],
        timeframe="1h",
        db_session=MagicMock(),
        config={"intrabar_stops": False},
    )
    assert bt_config.intrabar_stops is False


def test_cli_intrabar_stops_options():
    from click.testing import CliRunner
    from quantagent.cli.backtest import backtest_group

    runner = CliRunner()
    res = runner.invoke(backtest_group, ["run", "--help"])
    assert res.exit_code == 0
    assert "--intrabar-stops / --no-intrabar-stops" in res.output
