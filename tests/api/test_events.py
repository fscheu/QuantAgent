import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from apps.api.events import EVENT_TYPES, record_event
from quantagent.models import Event


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'events.db'}")
    Event.__table__.create(engine)
    return engine, Session(engine)


def test_unknown_type_is_rejected(tmp_path):
    engine, session = _session(tmp_path)
    try:
        with pytest.raises(ValueError, match="Unknown event type"):
            record_event(session, "not_an_event", "info", datetime.now(timezone.utc), None, None, {})
    finally:
        session.close()
        engine.dispose()


def test_insertion_order_and_payload_round_trip(tmp_path):
    engine, session = _session(tmp_path)
    payload = {"symbol": "MSFT", "details": {"quantity": 2, "tags": ["paper"]}}
    try:
        first_time = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
        second_time = datetime(2026, 10, 30, 12, 1, tzinfo=timezone.utc)
        first = record_event(session, "order_filled", "neutral", first_time, "paper-1", "order-1", payload)
        second = record_event(session, "position_closed", "neutral", second_time, "paper-1", "position-1", {})
        session.commit()

        rows = session.scalars(select(Event).order_by(Event.id)).all()
        assert [row.id for row in rows] == [first.id, second.id]
        assert rows[0].payload == payload
        assert rows[0].occurred_at == first_time.replace(tzinfo=None)
    finally:
        session.close()
        engine.dispose()


def test_catalog_matches_eventos_and_feed_event_types():
    root = Path(__file__).resolve().parents[2]
    eventos = (root / "docs/03_design/web-design-system/EVENTOS.md").read_text()
    catalog = re.search(r"^## 1\. Feed: catálogo de tipos$(.*?)(?=^## |\Z)", eventos, re.MULTILINE | re.DOTALL).group(1)
    table_types = set(re.findall(r"^\| `([^`]+)` \|", catalog, re.MULTILINE))
    assert EVENT_TYPES == table_types
