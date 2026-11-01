"""Read-only account chart and attribution data."""

import json
import os
from pathlib import Path

from sqlalchemy import inspect, select

from quantagent.data.snapshot import SnapshotError, read_snapshot


def equity_extras(account, rows):
    dates = [row.snapshot_date.isoformat() for row in rows]
    values = [float(row.equity) for row in rows]
    shadow = {day: None for day in dates}
    raw = account.database_url.removeprefix("sqlite:///")
    shadow_file = Path(raw).with_suffix(".sombra.json")
    try:
        saved = json.loads(shadow_file.read_text())
        for day in dates:
            item = saved.get(day, {}).get(Path(raw).stem)
            if item is not None:
                shadow[day] = float(item["valor"])
    except (OSError, ValueError, TypeError, KeyError):
        pass

    benchmark = {day: None for day in dates}
    name = os.environ.get("QUANTAGENT_WEB_SNAPSHOT")
    if name and rows:
        try:
            prices = read_snapshot(name, "SPY")
            first_rows = prices.loc[prices.index.date == rows[0].snapshot_date, "close"]
            first = float(first_rows.iloc[0]) if not first_rows.empty else None
            if first:
                for row in rows:
                    found = prices.loc[prices.index.date == row.snapshot_date, "close"]
                    if not found.empty:
                        benchmark[row.snapshot_date.isoformat()] = values[0] * float(found.iloc[0]) / first
        except (SnapshotError, OSError, ValueError, KeyError):
            pass
    return benchmark, shadow


def sector_exposure(account, source, today):
    from quantagent.models import EquitySnapshot, Position

    positions = source.scalars(select(Position)).all()
    snapshot_name = os.environ.get("QUANTAGENT_WEB_SNAPSHOT")
    grouped = {}
    if snapshot_name:
        from quantagent.data.sp500 import sector
    for position in positions:
        unresolved = False
        try:
            name = sector(snapshot_name, position.symbol, today) if snapshot_name else None
        except (SnapshotError, OSError, ValueError, KeyError):
            unresolved = True
            name = None
        name = "Sin clasificar" if unresolved or not snapshot_name else name or "ETF"
        values = grouped.setdefault(name, [0.0, 0.0])
        side = 0 if position.side.value == "buy" else 1
        values[side] += abs(float(position.quantity)) * float(position.current_price)

    cash = 0.0
    equity = 0.0
    if inspect(source.get_bind()).has_table("equity_snapshots"):
        latest = source.scalar(select(EquitySnapshot).order_by(EquitySnapshot.snapshot_date.desc()).limit(1))
        cash = float(latest.cash) if latest else 0.0
        equity = float(latest.equity) if latest else 0.0
    grouped["Cash"] = [cash, 0.0]
    return [{"sector": name, "long": amounts[0], "short": amounts[1],
             "pct": (sum(amounts) / equity if equity else 0.0)} for name, amounts in sorted(grouped.items())]
