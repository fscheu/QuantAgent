"""Read Alpaca dividend activities for a trading day."""

from decimal import Decimal, InvalidOperation
from datetime import datetime, time, timezone

from sqlalchemy import select


def leer_dividendos_alpaca(client, day):
    activities = client.get("/account/activities/DIV", data={"date": day.isoformat()})
    dividends = []
    for item in activities:
        if not isinstance(item, dict) or not isinstance(item.get("symbol"), str) or not item["symbol"].strip():
            raise ValueError("Actividad DIV malformada: falta symbol válido")
        try:
            amount = Decimal(str(item.get("net_amount")))
            if not amount.is_finite():
                raise InvalidOperation
        except InvalidOperation as exc:
            raise ValueError("Actividad DIV malformada: net_amount debe ser numérico y finito") from exc
        dividends.append((item["symbol"], item["net_amount"], "USD", "efectivo en cuenta"))
    return dividends


def registrar_dividendos_evento(session, account_id, day, dividends):
    """Append one daily Alpaca dividend event unless that account/date exists."""
    from apps.api.events import record_event
    from quantagent.models import Event

    entity_id = f"alpaca:{day.isoformat()}"
    exists = session.scalar(select(Event.id).where(
        Event.type == "dividend_received", Event.account_id == account_id, Event.entity_id == entity_id))
    if exists is not None or not dividends:
        return False
    record_event(session, "dividend_received", "neutral",
                 datetime.combine(day, time.max, tzinfo=timezone.utc), account_id, entity_id,
                 {"symbols": sorted({item[0] for item in dividends}),
                  "net": sum(float(item[1]) for item in dividends), "withheld": 0})
    return True
