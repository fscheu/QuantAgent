"""CSV serialization for backtest trade logs (D11 of PLAN-30-DIAS)."""

import csv
import io
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

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
