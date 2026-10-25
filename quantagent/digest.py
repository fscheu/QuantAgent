"""Formatting for the daily paper trading digest."""

from datetime import date, datetime, timedelta


def build_digest(day: date, portfolios: list[dict], trades: list[dict],
                 heartbeats: list[dict], warnings: list[str] | None = None,
                 reconciliations: list[str] | None = None,
                 cumulative_pnl: float | None = None) -> str:
    """Build a plain-text digest from already queried paper trading data."""
    sections = [f"Paper trading digest — {day.isoformat()}"]
    for portfolio in portfolios:
        positions = portfolio.get("positions", [])
        lines = [f"## {portfolio['name']} positions"]
        lines.extend(f"- {p['symbol']}: {p['quantity']} @ {p['price']} (uPnL {p['unrealized_pnl']})" for p in positions)
        if not positions:
            lines.append("- No open positions")
        sections.append("\n".join(lines))
    trade_lines = [f"## Activity and P&L ({len(trades)} trades)"]
    trade_lines.extend(f"- {t['symbol']} {t['side']} {t['quantity']}: P&L {t.get('pnl', 0)}" for t in trades)
    trade_lines.append(f"Daily realized P&L: {sum(float(t.get('pnl') or 0) for t in trades):.2f}")
    total_pnl = cumulative_pnl if cumulative_pnl is not None else sum(float(t.get("pnl") or 0) for t in trades)
    trade_lines.append(f"Cumulative realized P&L: {total_pnl:.2f}")
    if not trades:
        trade_lines.append("No trading activity")
    sections.append("\n".join(trade_lines))
    sections.append("## Warnings and errors\n" + ("\n".join(f"- {item}" for item in warnings) if warnings else "- None"))
    heartbeat_lines = ["## Scheduler heartbeat"]
    for heartbeat in heartbeats:
        heartbeat_lines.append(f"- {heartbeat.get('timestamp')}: {heartbeat.get('status')}")
        if heartbeat.get("error_message"):
            heartbeat_lines.append(f"  Error: {heartbeat['error_message']}")
    if not heartbeats:
        heartbeat_lines.append("- No heartbeat")
    else:
        successful = [h["timestamp"].date() for h in heartbeats if h.get("status") == "completed" and h.get("timestamp")]
        latest = max(successful or [h["timestamp"].date() for h in heartbeats if h.get("timestamp")], default=None)
        today = day
        business_days = sum(1 for offset in range(1, (today - latest).days + 1) if (latest + timedelta(days=offset)).weekday() < 5) if latest else 0
        if business_days > 1:
            heartbeat_lines.append("- ALERT: no successful cycle in more than one business day")
    sections.append("\n".join(heartbeat_lines))
    if reconciliations:
        sections.append("## Reconciliation\n" + "\n".join(f"- {item}" for item in reconciliations))
    text = "\n\n".join(sections)
    if len(text) >= 4000:
        marker = "\n\n[Digest truncated]"
        limit = 3999 - len(marker)
        kept = []
        for section in sections:
            candidate = "\n\n".join(kept + [section])
            if len(candidate) > limit:
                break
            kept.append(section)
        return "\n\n".join(kept) + marker
    return text
