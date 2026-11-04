"""Daily digest for the M5 paper trading bucket."""

from datetime import date, datetime, time, timedelta

from sqlalchemy.orm import Session

from quantagent.digest import SEPARADOR_MENSAJES, dividir_seccion
from quantagent.models import Environment, Order, OrderSide, OrderStatus, OrderType, Position, SchedulerHeartbeat, TradeIntentRecord
from quantagent.trading.bucket_m5 import CAPITAL_INICIAL, MAX_HEAT_PCT


def consultar_datos_m5(session: Session, day: date) -> dict:
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    intents = (session.query(TradeIntentRecord)
               .filter(TradeIntentRecord.bucket_id == "m5", TradeIntentRecord.created_at >= start,
                       TradeIntentRecord.created_at < end)
               .order_by(TradeIntentRecord.created_at, TradeIntentRecord.id).all())
    heartbeats = (session.query(SchedulerHeartbeat)
                  .filter(SchedulerHeartbeat.environment == Environment.PAPER,
                          SchedulerHeartbeat.timestamp >= start, SchedulerHeartbeat.timestamp < end)
                  .order_by(SchedulerHeartbeat.timestamp).all())
    ciclo = [{"status": item.status, "error_message": item.error_message,
              "modo": (item.stats or {}).get("modo")} for item in heartbeats]
    approved = [intent for intent in intents if intent.status == "aprobado"]
    rejected = [intent for intent in intents if intent.status == "rechazado_riesgo"]
    aprobados = []
    for intent in approved:
        order = session.get(Order, intent.order_id) if intent.order_id else None
        aprobados.append({"symbol": intent.symbol, "qty": float(order.quantity) if order else 0.0,
                          "entry": intent.intent["entry_price"], "stop": intent.intent["stop_loss"]})

    stops_por_simbolo = {}
    for intent in (session.query(TradeIntentRecord)
                   .filter(TradeIntentRecord.bucket_id == "m5", TradeIntentRecord.status == "aprobado")
                   .order_by(TradeIntentRecord.created_at.desc(), TradeIntentRecord.id.desc()).all()):
        stops_por_simbolo.setdefault(intent.symbol, intent.intent["stop_loss"])

    posiciones = []
    for position in session.query(Position).filter(Position.quantity > 0).order_by(Position.symbol).all():
        stop_order = (session.query(Order).filter(
            Order.symbol == position.symbol, Order.side == OrderSide.SELL, Order.order_type == OrderType.STOP,
            Order.status.notin_([OrderStatus.CANCELLED, OrderStatus.FILLED]), Order.stop_price.isnot(None)
        ).order_by(Order.created_at.desc(), Order.id.desc()).first())
        stop = stop_order.stop_price if stop_order else stops_por_simbolo.get(position.symbol)
        posiciones.append({"symbol": position.symbol, "quantity": float(position.quantity),
                           "avg_entry": float(position.average_entry_price), "unrealized_pnl": float(position.unrealized_pnl),
                           "stop": float(stop) if stop is not None else None})
    return {"aprobados": aprobados, "ciclo": ciclo,
            "rechazados": [{"symbol": item.symbol, "reject_reason": item.reject_reason} for item in rejected],
            "posiciones": posiciones}


def build_digest_m5_mensajes(day: date, datos: dict, *, capital: float = CAPITAL_INICIAL,
                    max_heat: float = MAX_HEAT_PCT) -> list[str]:
    approved = datos["aprobados"]
    rejected = datos["rechazados"]
    positions = datos["posiciones"]
    alertas = []
    ciclos_diarios = [ciclo for ciclo in datos.get("ciclo", []) if ciclo.get("modo") == "diario"]
    if day.weekday() < 5 and not any(ciclo.get("status") == "completed" for ciclo in ciclos_diarios):
        fallidos = [ciclo for ciclo in ciclos_diarios if ciclo.get("status") == "error"]
        corriendo = [ciclo for ciclo in ciclos_diarios if ciclo.get("status") == "running"]
        if fallidos:
            alertas.append(f"ALERTA: el ciclo diario M5 falló: {fallidos[-1].get('error_message') or 'error sin detalle'}")
        elif corriendo:
            alertas.append("ALERTA: el ciclo diario M5 sigue en running")
        else:
            alertas.append("ALERTA: el ciclo diario M5 no terminó OK (sin heartbeat completed)")
    sections = []
    lines = [f"Intents aprobados ({len(approved)})"]
    lines.extend(f"- {item['symbol']} {format(item['qty'], '.4f').rstrip('0').rstrip('.')} @ {float(item['entry']):.2f}, stop {float(item['stop']):.2f}"
                 for item in approved)
    sections.append("\n".join(lines))
    lines = [f"Intents rechazados ({len(rejected)})"]
    lines.extend(f"- {item['symbol']}: {item['reject_reason']}" for item in rejected)
    if not approved and not rejected:
        lines.append("Sin actividad de intents hoy.")
    sections.append("\n".join(lines))
    lines = ["Stops vigentes"]
    lines.extend(f"- {position['symbol']}: " +
                 (f"{position['stop']:.2f}" if position["stop"] is not None else "sin stop")
                 for position in positions)
    if not positions:
        lines.append("- Sin posiciones abiertas")
    heat = sum(position["quantity"] * (position["avg_entry"] - position["stop"])
               for position in positions
               if position["stop"] is not None and position["stop"] < position["avg_entry"])
    equity = capital + sum(position["unrealized_pnl"] for position in positions)
    heat_pct = heat / equity * 100 if equity else 0.0
    heat_line = (f"Heat: {heat_pct:.2f} % (límite {max_heat * 100:.2f} %)")
    sections.append("\n".join(lines))
    lines = []
    if positions:
        lines.append("Posiciones")
        lines.extend(f"- {p['symbol']}: {format(p['quantity'], '.4f').rstrip('0').rstrip('.')}"
                     for p in positions)
    if lines:
        sections.append("\n".join(lines))
    if datos.get("semanal"):
        sections.append("\n".join(datos["semanal"]))
    summary = [*alertas, f"Resumen M5 — {day.isoformat()}", f"Equity: {equity:.2f} USD",
               "Caja: no disponible", "P&L del día realizado: sin datos",
               f"Intents aprobados: {len(approved)}; rechazados: {len(rejected)}",
               f"Posiciones: {len(positions)}", heat_line,
               f"Avisos: {sum(p['stop'] is None for p in positions)} posiciones sin stop"]
    return ["\n".join(summary), *[message for section in sections for message in dividir_seccion(section)]]


def build_digest_m5(day: date, datos: dict, *, capital: float = CAPITAL_INICIAL,
                    max_heat: float = MAX_HEAT_PCT) -> str:
    return SEPARADOR_MENSAJES.join(build_digest_m5_mensajes(day, datos, capital=capital, max_heat=max_heat))
