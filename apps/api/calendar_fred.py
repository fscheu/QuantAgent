"""Synchronize FRED releases and manually maintained calendar events."""

import os
import logging
from datetime import date
from pathlib import Path

import requests
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from quantagent.models import CalendarEvent

FRED_RELEASES = {
    10: "IPC",
    50: "Empleo (nonfarm payrolls)",
    54: "PCE",
}
FRED_BASE_URL = "https://api.stlouisfed.org/fred/"
logger = logging.getLogger(__name__)


def fred_release_fetcher(release_id):
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("falta FRED_API_KEY para sincronizar el calendario de FRED")
    try:
        response = requests.get(
            FRED_BASE_URL + "release/dates",
            params={
                "api_key": key,
                "file_type": "json",
                "release_id": release_id,
                "realtime_start": date.today().isoformat(),
                "realtime_end": "9999-12-31",
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["release_dates"]
    except requests.RequestException as exc:
        raise RuntimeError(str(exc).replace(key, "***")) from None


def sync_fred_releases(session: Session, fetcher):
    """Upsert upcoming dates for the confirmed FRED release allowlist."""
    failures = []
    count = 0
    for release_id, title in FRED_RELEASES.items():
        try:
            rows = fetcher(release_id)
            for row in rows:
                day = date.fromisoformat(row["date"])
                event = session.query(CalendarEvent).filter_by(
                    date=day, kind="macro", source="fred"
                ).one_or_none()
                if event is None:
                    session.add(CalendarEvent(date=day, kind="macro", title=title, source="fred"))
                else:
                    event.title = title
            session.commit()
            count += len(rows)
        except Exception as exc:
            session.rollback()
            failures.append((release_id, exc))
            logger.exception("No se pudo sincronizar release FRED %s", release_id)
    if len(failures) == len(FRED_RELEASES):
        raise RuntimeError("fallaron todos los releases de FRED") from failures[-1][1]
    return count


def sync_manual_events(session: Session, path):
    """Replace the manual calendar source with validated YAML events."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("el YAML manual debe ser un objeto")
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError("events debe ser una lista")
    normalized = []
    for item in events:
        try:
            day = date.fromisoformat(item["date"])
            if day.isoformat() != item["date"]:
                raise ValueError
            title = item["title"]
            if not isinstance(title, str) or not title.strip():
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"evento manual inválido: {item}") from exc
        normalized.append((day, title.strip()))

    existing = session.query(CalendarEvent).filter_by(source="manual").all()
    expected = {(day, "macro", "manual") for day, _ in normalized}
    for event in existing:
        if (event.date, event.kind, event.source) not in expected:
            session.delete(event)
    for day, title in normalized:
        event = session.query(CalendarEvent).filter_by(date=day, kind="macro", source="manual").one_or_none()
        if event is None:
            session.add(CalendarEvent(date=day, kind="macro", title=title, source="manual"))
        else:
            event.title = title
    session.commit()
    return len(normalized)


def sync_fred_calendar():
    engine = create_engine(os.environ["QUANTAGENT_DATABASE_URL"])
    manual_path = Path(__file__).resolve().parents[2] / "calendario" / "eventos_manuales.yaml"
    with Session(engine) as session:
        fred_error = None
        try:
            fred_count = sync_fred_releases(session, fred_release_fetcher)
        except RuntimeError as exc:
            fred_count, fred_error = 0, exc
        manual_count = sync_manual_events(session, manual_path)
        if fred_error:
            raise fred_error
        return fred_count + manual_count
