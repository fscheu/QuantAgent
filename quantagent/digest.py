"""Formatting for the daily paper trading digest."""

from datetime import date, datetime, timedelta


def build_digest(day: date, portfolios: list[dict], trades: list[dict],
                 heartbeats: list[dict], warnings: list[str] | None = None,
                 reconciliations: list[str] | None = None,
                 cumulative_pnl: float | None = None, *,
                 valuation: dict | None = None, executions: list[dict] | None = None,
                 pending: list[dict] | None = None) -> str:
    """Build a plain-text digest from already queried paper trading data."""
    header = f"Resumen de paper trading — {day.isoformat()}"
    warning_items = [f"- {item}" for item in (warnings or [])]
    warning_text = "\n".join(warning_items)
    if len(warning_text) > 1000:
        warning_items = [item[:980] for item in warning_items[:1]]
        warning_items.append(f"... y {len(warnings) - 1 if warnings else 0} más")
        warning_text = "\n".join(warning_items)
    warning_section = "## Alertas y avisos\n" + (warning_text or "- Ninguno")
    heartbeat_lines = ["## Señal del scheduler"]
    for heartbeat in heartbeats[-3:]:
        heartbeat_lines.append(f"- {heartbeat.get('timestamp')}: {heartbeat.get('status')}")
        if heartbeat.get("error_message"):
            heartbeat_lines.append(f"  Error: {heartbeat['error_message']}")
    successful = [h["timestamp"].date() for h in heartbeats if h.get("status") == "completed" and h.get("timestamp")]
    latest = max(successful, default=None)
    business_days = sum(1 for offset in range(1, (day - latest).days + 1) if (latest + timedelta(days=offset)).weekday() < 5) if latest else 0
    if not heartbeats or latest is None or business_days > 1:
        heartbeat_lines.append("- ALERT: sin ciclo completado en más de un día hábil")
    if not heartbeats:
        heartbeat_lines.append("- Sin heartbeat")
    sections = [header, warning_section, "\n".join(heartbeat_lines)]
    if valuation is not None:
        portfolios = valuation["portfolios"]
        cash_pct = f"{100 * valuation['cash'] / valuation['equity']:.2f} %" if valuation["equity"] else "no disponible"
        sections.append(f"## Cuenta actual\nEquity: {valuation['equity']:.2f} USD\n"
                        f"Caja: {valuation['cash']:.2f} USD ({cash_pct}); reserva objetivo 5 %\n"
                        f"{valuation['notice']} Consultada: {valuation['consulted_at']}")
    if reconciliations:
        sections.append("## Reconciliación\n" + "\n".join(f"- {item}" for item in reconciliations))
    trade_items = [f"- {t['symbol']} {t['side']} {t['quantity']}: P&L {t.get('pnl', 0)}" for t in trades]
    trade_lines = [f"## Actividad y P&L ({len(trades)} operaciones)", *trade_items,
                   f"P&L realizado diario: {sum(float(t.get('pnl') or 0) for t in trades):.2f}"]
    total_pnl = cumulative_pnl if cumulative_pnl is not None else sum(float(t.get("pnl") or 0) for t in trades)
    trade_lines.append(f"P&L realizado acumulado: {total_pnl:.2f}")
    if not trades:
        trade_lines.append("Sin actividad de trading")
        if executions is not None and cumulative_pnl is None:
            trade_lines = ["## P&L realizado", "Sin datos de P&L realizado"]
    position_groups = []
    position_remaining = []
    for portfolio in portfolios:
        positions = portfolio.get("positions", [])
        lines = [f"## Posiciones de {portfolio['name']}"]
        lines.extend(f"- {p['symbol']}: {p['quantity']} @ {p['price']} "
                     + (f"(valor de mercado {p['value']:.2f} USD) " if valuation is not None else "")
                     + f"(P&L no realizado {p['unrealized_pnl']})" for p in positions)
        if not positions:
            lines.append("- Sin posiciones abiertas")
        position_groups.append(lines)
        position_remaining.append(len(positions))
    for title, items in [
        ("Ejecuciones del día", None if executions is None else [
            f"- {e['filled_at']}: {e['symbol']} {e['side']} {e['quantity']} @ {e['price']}; comisión {e['commission']}"
            for e in executions]),
        ("Órdenes pendientes", None if pending is None else [
            f"- {o['symbol']} {o['side']}: {o['status']}, {o['filled_quantity']}/{o['quantity']} ejecutadas"
            for o in pending]),
    ]:
        if items is not None:
            position_groups.append([f"## {title} ({len(items)})", *(items or ["- Ninguna"])])
            position_remaining.append(len(items))
    trade_index = len(sections)
    sections.append("\n".join(trade_lines))
    position_indexes = []
    for group in position_groups:
        position_indexes.append(len(sections))
        sections.append("\n".join(group))
    marker = "\n\n[Resumen truncado]"
    truncated_positions = [0] * len(position_groups)
    truncated_trades = 0
    text = "\n\n".join(sections)
    while len(text) >= 4000:
        if any(count > 0 for count in position_remaining):
            group_index = next(i for i in reversed(range(len(position_groups))) if position_remaining[i] > 0)
            group = position_groups[group_index]
            group.pop(-2 if group[-1].startswith("... y ") else -1)
            position_remaining[group_index] -= 1
            truncated_positions[group_index] += 1
            if group[-1].startswith("... y "):
                group[-1] = f"... y {truncated_positions[group_index]} más"
            else:
                group.append(f"... y {truncated_positions[group_index]} más")
            sections[position_indexes[group_index]] = "\n".join(group)
        elif len(trade_items) > 0:
            trade_items.pop()
            truncated_trades += 1
            trade_lines = [f"## Actividad y P&L ({len(trades)} operaciones)", *trade_items,
                           f"... y {truncated_trades} más",
                           f"P&L realizado diario: {sum(float(t.get('pnl') or 0) for t in trades):.2f}",
                           f"P&L realizado acumulado: {total_pnl:.2f}"]
            sections[trade_index] = "\n".join(trade_lines)
        else:
            break
        text = "\n\n".join(sections) + marker
        if len(text) < 4000:
            return text
    return text
