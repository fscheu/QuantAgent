"""Tests for quantagent.backtesting.export.trades_to_csv (D12 of PLAN-30-DIAS)."""

import csv
import io
from datetime import datetime

from quantagent.backtesting.export import COLUMNS, TradeRow, trades_to_csv


def _row(**overrides):
    defaults = dict(
        entry_time=datetime(2026, 4, 2, 14, 0, 0),
        exit_time=datetime(2026, 4, 3, 9, 0, 0),
        symbol="SPY",
        side="long",
        qty=12,
        entry_price=512.40,
        exit_price=508.15,
        stop_loss=502.15,
        pnl=-51.00,
        exit_reason="stop_loss",
    )
    defaults.update(overrides)
    return TradeRow(**defaults)


def test_header_matches_entregable_columns_exactly():
    csv_text = trades_to_csv([])
    header = next(csv.reader(io.StringIO(csv_text)))
    assert header == [
        "entry_time",
        "exit_time",
        "symbol",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "stop_loss",
        "pnl",
        "exit_reason",
    ]
    assert header == COLUMNS


def test_empty_trade_list_produces_only_header():
    csv_text = trades_to_csv([])
    lines = csv_text.strip("\r\n").split("\r\n")
    assert len(lines) == 1
    assert lines[0] == ",".join(COLUMNS)


def test_one_row_per_trade_with_formatted_values():
    csv_text = trades_to_csv([_row()])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert len(rows) == 2  # header + 1 trade

    data = dict(zip(rows[0], rows[1]))
    assert data["entry_time"] == "2026-04-02T14:00:00"
    assert data["exit_time"] == "2026-04-03T09:00:00"
    assert data["symbol"] == "SPY"
    assert data["side"] == "long"
    assert data["qty"] == "12"
    assert data["entry_price"] == "512.4"
    assert data["exit_price"] == "508.15"
    assert data["stop_loss"] == "502.15"
    assert data["pnl"] == "-51.0"
    assert data["exit_reason"] == "stop_loss"


def test_none_values_serialize_as_empty_string_not_the_word_none():
    csv_text = trades_to_csv(
        [_row(exit_time=None, exit_price=None, stop_loss=None, pnl=None, exit_reason=None)]
    )
    rows = list(csv.reader(io.StringIO(csv_text)))
    data = dict(zip(rows[0], rows[1]))
    assert data["exit_time"] == ""
    assert data["exit_price"] == ""
    assert data["stop_loss"] == ""
    assert data["pnl"] == ""
    assert data["exit_reason"] == ""
    assert "None" not in csv_text


def test_multiple_trades_preserve_order():
    row_a = _row(symbol="SPY", entry_price=1)
    row_b = _row(symbol="QQQ", entry_price=2)
    csv_text = trades_to_csv([row_a, row_b])
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert [r[2] for r in rows[1:]] == ["SPY", "QQQ"]


def test_csv_text_is_valid_utf8():
    csv_text = trades_to_csv([_row(exit_reason="stöp_löss")])
    encoded = csv_text.encode("utf-8")
    assert encoded.decode("utf-8") == csv_text
