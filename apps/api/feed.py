"""Build the read-only activity feed from master-database events.

Payload keys used by templates (missing keys are omitted gracefully):
- drawdown_limit: drawdown_pct, limit_pct
- stop_triggered: symbol, side, quantity, price, pnl
- order_rejected: reason
- stop_near: symbol, side, pct, stop, price, direction
- stop_breached: symbol, stop, price
- shadow_divergence: gap, date
- data_stale: duration, last_seen
- alarm_resolved: title, duration_seconds, alarm_type
- hypothesis_promoted: name, stage, metric
- hypothesis_discarded_late: name, stage, reason
- cycle_completed: cycle, entered, remaining, dropped
- order_filled: symbol, side, quantity, price
- position_closed: symbol, pnl, pct
- dividend_received: symbols, net, withheld

Alarm deduplication keeps only the first occurrence of each active alarm key in
the requested window. If a later alarm_resolved event matches account_id,
entity_id, and payload.alarm_type, the alarm is resolved and both its first
occurrence and resolution are shown.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select

from quantagent.models import Account, Event

NEW_YORK = ZoneInfo("America/New_York")
ALARM_TYPES = {"drawdown_limit", "stop_near", "stop_breached", "shadow_divergence", "data_stale"}
ENTITY_TYPES = {"stop_near", "stop_breached", "alarm_resolved", "hypothesis_promoted",
                "hypothesis_discarded_late", "cycle_completed"}
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2, "neutral": 3}


def _value(payload, key):
    value = payload.get(key) if isinstance(payload, dict) else None
    return value if value is not None and value != "" else None


def _number(value, suffix=""):
    if value is None:
        return None
    try:
        result = f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        result = result.rstrip("0").rstrip(",") if "," in result else result
        return result + suffix
    except (TypeError, ValueError):
        return str(value) + suffix


def _template(event, account_name, group=None):
    group = group or [event]
    p = event.payload or {}
    typ = event.type
    account = account_name or event.account_id or "la cuenta"
    symbol, side = _value(p, "symbol"), _value(p, "side")
    title, subtitle = typ.replace("_", " ").capitalize(), None
    if typ == "drawdown_limit":
        title = f"Drawdown sobre el límite en {account}"
        bits = [f"{_number(_value(p, 'drawdown_pct'), ' %')} desde el máximo" if _value(p, 'drawdown_pct') is not None else None,
                f"límite {_number(_value(p, 'limit_pct'), ' %')}" if _value(p, 'limit_pct') is not None else None]
        subtitle = "; ".join(x for x in bits if x) or None
    elif typ == "stop_triggered":
        title = (f"Saltó el stop de {symbol}" if symbol else f"Saltó un stop en {account}") if len(group) == 1 else f"Saltaron {len(group)} stops en {account}"
        action = "Vendidas" if side in ("long", "buy") else "Recompradas" if side in ("short", "sell") else None
        q, price, pnl = (_value(p, k) for k in ("quantity", "price", "pnl"))
        subtitle = " · ".join(x for x in [f"{action} {_number(q)} a {_number(price)}" if action and q is not None and price is not None else None,
                                           f"P&L {_number(pnl)}" if pnl is not None else None] if x) or None
    elif typ == "order_rejected":
        count = len(group)
        title = f"1 orden rechazada en {account}" if count == 1 else f"{count} órdenes rechazadas en {account}"
        reasons = [str(r) for r in (_value(item.payload or {}, "reason") for item in group) if r is not None]
        subtitle = ", ".join(dict.fromkeys(reasons)) or None
    elif typ == "stop_near":
        title = " ".join(x for x in [symbol, f"({side})" if side else None, "quedó a", _number(_value(p, "pct"), " %"), "del stop"] if x) or title
        subtitle = " · ".join(x for x in [f"{account}", f"stop {_value(p, 'direction')} {_number(_value(p, 'stop'))}" if _value(p, "stop") is not None else None,
                                           f"hoy {_number(_value(p, 'price'))}" if _value(p, "price") is not None else None] if x) or None
    elif typ == "stop_breached":
        title = f"{symbol}: el precio pasó el stop" if symbol else "El precio pasó el stop"
        subtitle = " · ".join(x for x in [f"Stop {_number(_value(p, 'stop'))}" if _value(p, "stop") is not None else None,
                                           f"hoy {_number(_value(p, 'price'))}" if _value(p, "price") is not None else None, "cierre pendiente"] if x) or None
    elif typ == "shadow_divergence":
        title = f"Sombra desalineada en {account}"
        subtitle = " · ".join(x for x in [f"Difieren {_number(_value(p, 'gap'))}" if _value(p, "gap") is not None else None,
                                           f"desde {_value(p, 'date')}" if _value(p, "date") is not None else None] if x) or None
    elif typ == "data_stale":
        title = f"Sin datos de {account}" + (f" desde hace {_value(p, 'duration')}" if _value(p, "duration") is not None else "")
        subtitle = f"Último dato {_value(p, 'last_seen')} ART" if _value(p, "last_seen") is not None else None
    elif typ == "alarm_resolved":
        title = f"Se resolvió: {_value(p, 'title')}" if _value(p, "title") is not None else "Se resolvió una alarma"
        d = _value(p, "duration_seconds")
        subtitle = f"Duró {_number(float(d) / 60 if isinstance(d, (int, float)) else d)} min" if d is not None else None
    elif typ in ("hypothesis_promoted", "hypothesis_discarded_late"):
        name, stage = _value(p, "name"), _value(p, "stage")
        if typ == "hypothesis_promoted":
            title = f"{name} pasó a {stage}" if name and stage else "Una hipótesis avanzó"
            detail = _value(p, "metric")
        else:
            title = f"Se descartó {name} en {stage}" if name and stage else "Se descartó una hipótesis"
            detail = _value(p, "reason")
        subtitle = str(detail) if detail is not None else None
    elif typ == "cycle_completed":
        cycle = _value(p, "cycle")
        title = f"Terminó el ciclo {cycle} del embudo" if cycle is not None else "Terminó un ciclo del embudo"
        vals = [_value(p, k) for k in ("entered", "remaining", "dropped")]
        if any(v is not None for v in vals):
            subtitle = ", ".join(f"{_number(v)} {label}" for v, label in zip(vals, ("entraron", "siguen", "cayeron")) if v is not None)
    elif typ == "order_filled":
        sides = {_value(item.payload or {}, "side") for item in group}
        count = len(group)
        if sides == {"buy"}:
            title = f"{count} compras ejecutadas"
        elif sides == {"sell"}:
            title = f"{count} ventas ejecutadas"
        else:
            title = f"{count} órdenes ejecutadas"
        lines = []
        for item in group:
            payload = item.payload or {}
            parts = [_value(payload, "symbol"), _number(_value(payload, "quantity"))]
            price = _value(payload, "price")
            if price is not None:
                parts.append(f"a {_number(price)}")
            lines.append(" ".join(str(part) for part in parts if part))
        subtitle = " · ".join(line for line in lines if line) or None
    elif typ == "position_closed":
        title = f"Se cerró {symbol}" if symbol else "Se cerró una posición"
        subtitle = " · ".join(x for x in [f"P&L realizado {_number(_value(p, 'pnl'))}" if _value(p, "pnl") is not None else None,
                                           _number(_value(p, "pct"), " %")] if x) or None
    elif typ == "dividend_received":
        symbols = list(dict.fromkeys(symbol for item in group for symbol in (_value(item.payload or {}, "symbols") or []) if isinstance(symbol, str)))
        if not symbols:
            symbols = _value(p, "symbols")
        title = f"Cobraste dividendos de {', '.join(map(str, symbols)) if isinstance(symbols, list) else symbols}" if symbols else "Cobraste dividendos"
        subtitle = " · ".join(x for x in [f"Neto {_number(_value(p, 'net'))}" if _value(p, "net") is not None else None,
                                           f"retenido {_number(_value(p, 'withheld'))}" if _value(p, "withheld") is not None else None] if x) or None
    return title, subtitle


def _link(event):
    if not event.account_id:
        return "/"
    base = f"/cuenta/{event.account_id}"
    if event.type in {"stop_triggered", "order_rejected", "order_filled"}:
        return base + "?tab=ordenes"
    if event.type in {"stop_near", "stop_breached"}:
        return base + "?tab=posiciones"
    if event.type == "dividend_received":
        return base + "?tab=dividendos"
    if event.type in {"hypothesis_promoted", "hypothesis_discarded_late", "cycle_completed"}:
        return "/embudo"
    return base


def build_feed(db, since=None, limit=50, utc_z=None):
    """Query and group events into FeedEventResponse-compatible dictionaries."""
    since = since or datetime.now(timezone.utc) - timedelta(hours=24)
    rows = db.scalars(select(Event).where(Event.occurred_at >= since).order_by(Event.occurred_at, Event.id)).all()
    accounts = {a.id: a.name for a in db.scalars(select(Account)).all()}
    resolved = {(e.account_id, e.entity_id, (e.payload or {}).get("alarm_type")) for e in rows
                if e.type == "alarm_resolved"}
    first_alarm = {}
    visible = []
    for e in rows:
        if e.type in ALARM_TYPES:
            key = (e.type, e.account_id, e.entity_id)
            if key in first_alarm:
                continue
            first_alarm[key] = e
            if (e.account_id, e.entity_id, e.type) in resolved:
                visible.append(e)
            # Keep candidate; filter after all rows are inspected.
        else:
            visible.append(e)
    for key, e in first_alarm.items():
        if (e.account_id, e.entity_id, e.type) not in resolved:
            visible.append(e)
    groups = defaultdict(list)
    for e in visible:
        occurred = e.occurred_at
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=timezone.utc)
        day = occurred.astimezone(NEW_YORK).date()
        entity = e.entity_id if e.type in ENTITY_TYPES else None
        groups[(e.type, e.account_id, day, entity)].append(e)
    result = []
    for group in groups.values():
        latest = max(group, key=lambda e: (e.occurred_at, e.id))
        title, subtitle = _template(latest, accounts.get(latest.account_id), group)
        result.append({"id": str(latest.id), "type": latest.type, "severity": latest.severity,
            "occurred_at": utc_z(latest.occurred_at) if utc_z else latest.occurred_at.isoformat(),
            "account_id": latest.account_id, "title": title, "subtitle": subtitle,
            "link": _link(latest), "group_count": len(group)})
    # Severity rank asc, timestamp desc within severity.
    result.sort(key=lambda item: (SEVERITY_ORDER.get(item["severity"], 99), -datetime.fromisoformat(item["occurred_at"].replace("Z", "+00:00")).timestamp()))
    return result[:limit]
