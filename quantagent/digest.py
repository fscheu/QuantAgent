"""Formatting for the daily paper trading digest."""

from datetime import date, timedelta


LIMITE_MENSAJE = 3900
SEPARADOR_MENSAJES = "\n\n──────── Mensaje siguiente ────────\n\n"


def dividir_seccion(section: str) -> list[str]:
    """Split a section at line boundaries, preserving every character of its content."""
    messages = []
    current = ""
    for line in section.splitlines(keepends=True):
        while len(line) > LIMITE_MENSAJE:
            if current:
                messages.append(current)
                current = ""
            messages.append(line[:LIMITE_MENSAJE])
            line = line[LIMITE_MENSAJE:]
        if len(current) + len(line) > LIMITE_MENSAJE:
            messages.append(current)
            current = ""
        current += line
    if current:
        messages.append(current)
    return messages


def build_digest_mensajes(day: date, portfolios: list[dict], trades: list[dict],
                 heartbeats: list[dict], warnings: list[str] | None = None,
                 reconciliations: list[str] | None = None,
                 cumulative_pnl: float | None = None, *,
                 valuation: dict | None = None, executions: list[dict] | None = None,
                 pending: list[dict] | None = None) -> list[str]:
    """Build a plain-text digest from already queried paper trading data."""
    header = f"Resumen de paper trading — {day.isoformat()}"
    warning_items = [f"- {item}" for item in (warnings or [])]
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
    scheduler_alert = not heartbeats or latest is None or business_days > 1
    if scheduler_alert:
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
    for portfolio in portfolios:
        positions = portfolio.get("positions", [])
        lines = [f"## Posiciones de {portfolio['name']}"]
        lines.extend(f"- {p['symbol']}: {p.get('side', '') + ' ' if p.get('side') else ''}{p['quantity']} @ {p['price']} "
                     + (f"(valor de mercado {p['value']:.2f} USD) " if valuation is not None else "")
                     + f"(P&L no realizado {p['unrealized_pnl']})" for p in positions)
        if not positions:
            lines.append("- Sin posiciones abiertas")
        position_groups.append(lines)
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
    sections.append("\n".join(trade_lines))
    sections.extend("\n".join(group) for group in position_groups)
    daily_pnl = ("sin datos" if not trades and executions is not None and cumulative_pnl is None
                 else f"{sum(float(t.get('pnl') or 0) for t in trades):.2f} USD")
    summary = [header,
               f"Equity: {valuation['equity']:.2f} USD" if valuation else "Equity: no disponible (sin caja en DB)",
               f"Caja: {valuation['cash']:.2f} USD" if valuation else "Caja: no disponible",
               f"P&L del día realizado: {daily_pnl}",
               f"Desvíos de reconciliación: {len(reconciliations or [])}",
               f"Órdenes pendientes: {len(pending)}" if pending is not None else "Órdenes pendientes: sin datos",
               f"Alertas y avisos: {len(warnings or [])}; scheduler: {'ALERT' if scheduler_alert else 'con ciclo completado'}",
               f"Ejecuciones del día: {len(executions)}" if executions is not None else f"Operaciones del día: {len(trades)}"]
    return ["\n".join(summary), *[message for section in sections[1:] for message in dividir_seccion(section)]]


def build_digest(day: date, portfolios: list[dict], trades: list[dict],
                 heartbeats: list[dict], warnings: list[str] | None = None,
                 reconciliations: list[str] | None = None,
                 cumulative_pnl: float | None = None, *,
                 valuation: dict | None = None, executions: list[dict] | None = None,
                 pending: list[dict] | None = None) -> str:
    """Build the human-readable digest with separators between messages."""
    return SEPARADOR_MENSAJES.join(build_digest_mensajes(
        day, portfolios, trades, heartbeats, warnings, reconciliations, cumulative_pnl,
        valuation=valuation, executions=executions, pending=pending))
