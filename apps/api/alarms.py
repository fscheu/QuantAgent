"""Read-only evaluation of account alarms."""

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from sqlalchemy import inspect, select

from quantagent.models import CalendarEvent, EquitySnapshot, Event, Position

from .accounts import account_session_factory
from .alarm_config import AlarmConfig
from .positions import read_positions

NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class AlarmResponse(BaseModel):
    id: str
    account_id: str
    severity: str
    kind: str
    title: str
    message: str
    since: str
    payload: dict = Field(exclude=True)


def _aware(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _last_open_event(master_db, account_id, kind, now):
    if not inspect(master_db.get_bind()).has_table("events"):
        return now
    events = master_db.scalars(select(Event).where(Event.account_id == account_id,
        Event.type.in_([kind, "alarm_resolved"])).order_by(Event.occurred_at, Event.id)).all()
    opened = None
    for event in events:
        if event.type == kind:
            opened = event
        elif (event.payload or {}).get("alarm_type") == kind:
            opened = None
    return _aware(opened.occurred_at) if opened else now


def evaluate_account_alarms(account, master_db, now, config=None):
    config = config or AlarmConfig.from_env()
    now = _aware(now).astimezone(UTC)
    alarms = []
    with account_session_factory(account)() as db:
        tables = inspect(db.get_bind())
        snapshots = db.scalars(select(EquitySnapshot).order_by(EquitySnapshot.snapshot_date)).all() if tables.has_table("equity_snapshots") else []
        positions = db.scalars(select(Position)).all() if tables.has_table("positions") else []
        if len(snapshots) >= 2:
            peak = max(float(row.equity) for row in snapshots)
            latest = snapshots[-1]
            drawdown = (peak - float(latest.equity)) / peak * 100 if peak > 0 else 0
            if drawdown > config.drawdown_limit_pct:
                alarms.append(("drawdown_limit", "critical", f"Drawdown sobre el límite", f"Drawdown {drawdown:.2f} %; límite {config.drawdown_limit_pct:g} %", {"drawdown_pct": drawdown, "limit_pct": config.drawdown_limit_pct}))

        database = account.database_url
        sqlite_file = database.startswith("sqlite:////") or database.startswith("sqlite:///")
        path = database.removeprefix("sqlite:///") if sqlite_file else None
        holdings = read_positions(db, "prod" if account.env == "live" else "paper")
        valued = [p for p in holdings if p.market_value is not None]
        if path and Path(path).is_file() and valued and all(p.current_price is not None for p in holdings):
            shadow_path = Path(path).with_suffix(".sombra.json")
            try:
                raw = json.loads(shadow_path.read_text())
                shadow_date = max(raw)
                shadow = raw[shadow_date][Path(path).stem]["pesos"]
                values = {p.symbol: p.market_value * (1 if p.side == "long" else -1) for p in valued}
                total = float(snapshots[-1].equity) if snapshots else sum(abs(value) for value in values.values())
                if total > 0:
                    actual = {symbol: value / total for symbol, value in values.items()}
                    gap = max((abs(actual.get(symbol, 0) - float(shadow.get(symbol, 0))) for symbol in set(actual) | set(shadow)), default=0)
                    if gap > config.shadow_gap:
                        alarms.append(("shadow_divergence", "warning", "Sombra desalineada", f"Diferencia máxima {gap:.2%}; límite {config.shadow_gap:.0%}", {"gap": gap, "date": shadow_date}))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass

        timestamps = [p.updated_at for p in positions] + [row.taken_at for row in snapshots]
        if timestamps:
            last_seen = max(_aware(value) for value in timestamps)
            local = now.astimezone(NEW_YORK)
            day = last_seen.astimezone(NEW_YORK).date()
            holidays = (set(master_db.scalars(select(CalendarEvent.date).where(CalendarEvent.kind == "holiday",
                CalendarEvent.date >= day, CalendarEvent.date <= local.date())).all())
                if inspect(master_db.get_bind()).has_table("calendar_events") else set())
            market_open = local.weekday() < 5 and time(9, 30) <= local.time().replace(tzinfo=None) < time(16) and local.date() not in holidays
            session_since_data = False
            while day <= local.date():
                opening = datetime.combine(day, time(9, 30), tzinfo=NEW_YORK)
                if day.weekday() < 5 and day not in holidays and last_seen < opening <= now:
                    session_since_data = True
                    break
                day += timedelta(days=1)
            threshold = config.stale_minutes_market_open if market_open else config.stale_minutes_market_closed
            age = (now - last_seen.astimezone(UTC)).total_seconds() / 60
            if age > threshold and (market_open or session_since_data):
                duration = f"{age / 60:.1f} horas" if age >= 60 else f"{age:.0f} minutos"
                alarms.append(("data_stale", "warning", "Sin datos recientes", f"Último dato hace {duration} ({last_seen.isoformat()})", {"duration": duration, "last_seen": last_seen.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M")}))

    return [AlarmResponse(id=f"{kind}:{account.id}", account_id=account.id, severity=severity, kind=kind,
        title=title, message=message, since=_last_open_event(master_db, account.id, kind, now).isoformat(), payload=payload)
        for kind, severity, title, message, payload in alarms]
