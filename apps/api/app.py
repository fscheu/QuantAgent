"""FastAPI application and read-only database dependency."""

import os
from datetime import date as Date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Query
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from .schemas import AccountRef, AccountSummaryActive, AccountSummaryNoData, OrderResponse, PositionResponse

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


def configured_account(account_id, db):
    from quantagent.models import Account

    account = db.get(Account, account_id)
    if account is None or account.status != "active":
        raise HTTPException(status_code=404, detail="Account not found")
    return account


def utc_z(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo("UTC"))
    return value.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")


@app.get("/api/accounts/{account_id}/summary", response_model=AccountSummaryActive | AccountSummaryNoData)
def account_summary(account_id: str, db: Session = Depends(get_db)):
    from quantagent.models import Order, Position
    from .accounts import account_session_factory

    account = configured_account(account_id, db)
    with account_session_factory(account)() as source:
        positions = source.scalars(select(Position).order_by(Position.symbol)).all()
        checked_at = utc_z(datetime.now(ZoneInfo("UTC")))
        if not positions:
            return AccountSummaryNoData(id=account.id, name=account.name, env=account.env,
                                        status="no_data", checked_at=checked_at)
        now_ny = datetime.now(NEW_YORK)
        day_start = now_ny.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        orders_today = source.scalar(select(func.count()).select_from(Order).where(Order.created_at >= day_start)) or 0
        rejected = source.scalar(select(func.count()).select_from(Order).where(
            Order.created_at >= day_start, Order.status == "rejected")) or 0
        long_value = sum(abs(float(p.quantity)) * float(p.current_price) for p in positions if p.side.value == "buy")
        short_value = sum(abs(float(p.quantity)) * float(p.current_price) for p in positions if p.side.value == "sell")
        as_of = utc_z(max(p.updated_at for p in positions))
        return AccountSummaryActive(id=account.id, name=account.name, env=account.env, status="active",
            as_of=as_of, stale=False, equity=None, long_value=long_value, short_value=short_value,
            cash=None, day_change=None, day_change_pct=None, week_change=None, week_change_pct=None,
            positions_count=len(positions), orders_today=orders_today, orders_rejected_today=rejected, alarms=[])


@app.get("/api/accounts/{account_id}/positions", response_model=list[PositionResponse])
def account_positions(account_id: str, db: Session = Depends(get_db)):
    from quantagent.models import ActivePosition, Position
    from .accounts import account_session_factory

    account = configured_account(account_id, db)
    with account_session_factory(account)() as source:
        active_columns = (ActivePosition.symbol, ActivePosition.side, ActivePosition.stop_loss,
                          ActivePosition.environment, ActivePosition.is_active)
        rows = source.execute(select(Position, *active_columns).outerjoin(ActivePosition,
            (ActivePosition.symbol == Position.symbol) & (ActivePosition.side == Position.side) &
            (ActivePosition.environment == ("prod" if account.env == "live" else "paper")) &
            ActivePosition.is_active.is_(True)).order_by(Position.symbol)).all()
        result = []
        for row in rows:
            position = row[0]
            side = "long" if position.side.value == "buy" else "short"
            qty, price = abs(float(position.quantity)), float(position.current_price)
            stop_loss = row[3]
            stop = float(stop_loss) if stop_loss is not None else None
            result.append(PositionResponse(symbol=position.symbol, side=side, qty=qty,
                avg_entry_price=float(position.average_entry_price), current_price=price,
                market_value=qty * price, unrealized_pl=float(position.unrealized_pnl),
                unrealized_pl_pct=float(position.unrealized_pnl_pct), stop_price=stop,
                stop_distance=(price - stop if side == "long" else stop - price) if stop is not None else None,
                opened_at=utc_z(position.opened_at)))
        return result


@app.get("/api/accounts/{account_id}/orders", response_model=list[OrderResponse])
def account_orders(account_id: str, date: str | None = Query(None), db: Session = Depends(get_db)):
    from quantagent.models import Order, TradeIntentRecord
    from .accounts import account_session_factory
    from .reject_reasons import rejection_text

    account = configured_account(account_id, db)
    try:
        day = Date.fromisoformat(date) if date is not None else datetime.now(NEW_YORK).date()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid date") from exc
    start = datetime.combine(day, datetime.min.time(), NEW_YORK).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), NEW_YORK).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    with account_session_factory(account)() as source:
        rows = source.scalars(select(Order).where(Order.created_at >= start, Order.created_at < end)
                              .order_by(Order.created_at, Order.id)).all()
        reasons = {}
        if rows and inspect(source.get_bind()).has_table("trade_intents"):
            reasons = dict(source.execute(select(TradeIntentRecord.order_id, TradeIntentRecord.reject_reason)
                .where(TradeIntentRecord.order_id.in_([row.id for row in rows]))).all())
        return [OrderResponse(id=row.id, symbol=row.symbol, side=row.side.value, type=row.order_type.value,
            status=row.status.value, qty=float(row.quantity), filled_qty=float(row.filled_quantity),
            price=float(row.price) if row.price is not None else None,
            stop_price=float(row.stop_price) if row.stop_price is not None else None,
            avg_fill_price=float(row.average_fill_price) if row.average_fill_price is not None else None,
            created_at=utc_z(row.created_at), filled_at=utc_z(row.filled_at) if row.filled_at else None,
            reject_reason=(rejection_text(reasons[row.id]) if reasons.get(row.id) else None)
                if row.status.value == "rejected" else None) for row in rows]
