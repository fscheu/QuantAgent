"""FastAPI application and read-only database dependency."""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Query
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

app = FastAPI(title="QuantAgent API")
NEW_YORK = ZoneInfo("America/New_York")


class CalendarEventResponse(BaseModel):
    id: str
    kind: Literal["holiday", "early_close"]
    starts_at: str | None = None
    date: str
    title: str
    subtitle: str | None = None
    source: Literal["alpaca"]
    account_id: str | None = None


class AccountRef(BaseModel):
    id: str
    name: str
    env: Literal["paper", "live"]


def session_factory(url=None):
    url = url or os.environ["QUANTAGENT_DATABASE_URL"]
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def readonly_sqlite(connection, _):
            connection.execute("PRAGMA query_only=ON")
    else:
        @event.listens_for(engine, "begin")
        def readonly_transaction(connection):
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
    return sessionmaker(bind=engine, class_=Session, autoflush=False)


def get_db():
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/calendar", response_model=list[CalendarEventResponse])
def calendar(days: int = Query(30, ge=1, le=120), db: Session = Depends(get_db)):
    from quantagent.models import CalendarEvent

    today = datetime.now(NEW_YORK).date()
    events = db.scalars(
        select(CalendarEvent).where(
            CalendarEvent.date >= today,
            CalendarEvent.date < today + timedelta(days=days),
        ).order_by(CalendarEvent.date, CalendarEvent.kind)
    ).all()
    return [
        CalendarEventResponse(id=str(event.id), kind=event.kind, date=event.date.isoformat(),
                              title=event.title, source=event.source)
        for event in events
    ]


@app.get("/api/accounts", response_model=list[AccountRef])
def accounts(db: Session = Depends(get_db)):
    from quantagent.models import Account

    rows = db.scalars(select(Account).where(Account.status == "active").order_by(Account.id)).all()
    return [AccountRef(id=row.id, name=row.name, env=row.env) for row in rows]
