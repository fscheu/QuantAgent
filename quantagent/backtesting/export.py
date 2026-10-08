"""CSV serialization for backtest trade logs (D11 of PLAN-30-DIAS)."""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

COLUMNS = [
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

EQUITY_COLUMNS = ["timestamp", "equity", "cash", "positions_value", "drawdown_pct"]


@dataclass
class TradeRow:
    """One row of the trade log CSV. Callers build this from Trade + ActivePosition."""

    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float]
    pnl: Optional[float]
    exit_reason: Optional[str]


def trades_to_csv(trades: List[TradeRow]) -> str:
    """Serialize trade rows into the entregable's 10-column CSV format."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(COLUMNS)

    for trade in trades:
        writer.writerow(
            [
                trade.entry_time.isoformat() if trade.entry_time else "",
                trade.exit_time.isoformat() if trade.exit_time else "",
                trade.symbol,
                trade.side,
                trade.qty,
                trade.entry_price,
                trade.exit_price if trade.exit_price is not None else "",
                trade.stop_loss if trade.stop_loss is not None else "",
                trade.pnl if trade.pnl is not None else "",
                trade.exit_reason or "",
            ]
        )

    return buffer.getvalue()


def equity_to_csv(equity_curve: List[Dict]) -> str:
    """Serialize the engine's equity curve, one row per point.

    drawdown_pct is the fall from the running peak as a positive fraction (0.15 = 15%),
    the same unit as BacktestMetrics.max_drawdown, so max(drawdown_pct) equals it up to
    the rounding: the four numeric columns are written with at most 4 decimal places.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EQUITY_COLUMNS)

    peak = None
    for point in equity_curve:
        equity = point["equity"]
        peak = equity if peak is None else max(peak, equity)
        writer.writerow(
            [
                point["date"].isoformat(),
                round(equity, 4),
                round(point["cash"], 4),
                round(point["positions_value"], 4),
                round((peak - equity) / peak if peak else 0.0, 4),
            ]
        )

    return buffer.getvalue()
