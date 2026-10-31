"""Append-only event recording for the activity feed."""

from datetime import timezone

from quantagent.models import Event

EVENT_TYPES = frozenset(
    {
        "drawdown_limit",
        "stop_triggered",
        "order_rejected",
        "stop_near",
        "stop_breached",
        "shadow_divergence",
        "data_stale",
        "alarm_resolved",
        "hypothesis_promoted",
        "hypothesis_discarded_late",
        "cycle_completed",
        "order_filled",
        "position_closed",
        "dividend_received",
    }
)
EVENT_SEVERITIES = frozenset({"critical", "warning", "info", "neutral"})


def record_event(session, type, severity, occurred_at, account_id, entity_id, payload):
    """Validate and add one event to the caller's transaction."""
    if type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {type}")
    if severity not in EVENT_SEVERITIES:
        raise ValueError(f"Unknown event severity: {severity}")
    if occurred_at.tzinfo is None:
        raise ValueError("occurred_at must be timezone-aware UTC")

    event = Event(
        type=type,
        severity=severity,
        occurred_at=occurred_at.astimezone(timezone.utc),
        account_id=account_id,
        entity_id=entity_id,
        payload=payload,
    )
    session.add(event)
    return event
