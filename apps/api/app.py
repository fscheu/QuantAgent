"""FastAPI application and read-only database dependency."""

import os
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Query
from fastapi import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from .schemas import AccountRef, AccountSummaryActive, AccountSummaryNoData, EquityPoint, OrderResponse, PositionResponse

app = FastAPI(title="QuantAgent API")
NEW_YORK = ZoneInfo("America/New_York")


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path == "api" or path.startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


class CalendarEventResponse(BaseModel):
    id: str
    kind: Literal["holiday", "early_close", "rebalance"]
    starts_at: str | None = None
    date: str
    title: str
    subtitle: str | None = None
    source: Literal["alpaca", "strategy"]
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
    from quantagent.models import Account, CalendarEvent
    from .rebalances import proximos_rebalanceos

    today = datetime.now(NEW_YORK).date()
    events = db.scalars(
        select(CalendarEvent).where(
            CalendarEvent.date >= today,
            CalendarEvent.date < today + timedelta(days=days),
        ).order_by(CalendarEvent.date, CalendarEvent.kind)
    ).all()
    result = [
        CalendarEventResponse(id=str(event.id), kind=event.kind, date=event.date.isoformat(),
                              title=event.title, source=event.source,
                              subtitle=None, account_id=None)
        for event in events
    ]
    cuentas = db.scalars(select(Account).where(Account.status == "active").order_by(Account.id)).all()
    feriados = db.scalars(select(CalendarEvent.date).where(CalendarEvent.kind == "holiday")).all()
    from quantagent.fabrica.hipotesis import cargar_hipotesis
    carpeta = Path(__file__).resolve().parents[2] / "hipotesis"
    rebalanceos = proximos_rebalanceos(
        cuentas, today, days, feriados,
        lambda strategy: cargar_hipotesis(carpeta / f"{strategy}.yaml"),
    )
    result.extend(CalendarEventResponse(
        id=f"rebalance:{cuenta.id}:{fecha.isoformat()}", kind="rebalance",
        date=fecha.isoformat(), title="Rebalanceo", subtitle=cuenta.name,
        source="strategy", account_id=cuenta.id,
    ) for cuenta, fecha in rebalanceos)
    return sorted(result, key=lambda event: (event.date, event.kind))


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
    from quantagent.models import EquitySnapshot, Order, Position
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
        snapshots = []
        if inspect(source.get_bind()).has_table("equity_snapshots"):
            snapshots = source.scalars(select(EquitySnapshot).order_by(EquitySnapshot.snapshot_date.desc()).limit(30)).all()
        latest = snapshots[0] if snapshots else None
        previous = snapshots[1] if len(snapshots) > 1 else None
        week_base = next((point for point in snapshots if point.snapshot_date <= latest.snapshot_date - timedelta(days=7)), None) if latest else None
        day_change = float(latest.equity - previous.equity) if latest and previous else None
        week_change = float(latest.equity - week_base.equity) if latest and week_base else None
        spark_start = datetime.now(NEW_YORK).date() - timedelta(days=29)
        spark = [float(point.equity) for point in reversed(snapshots)
                 if point.snapshot_date >= spark_start] if latest else []
        return AccountSummaryActive(id=account.id, name=account.name, env=account.env, status="active",
            as_of=as_of, stale=False, equity=float(latest.equity) if latest else None, long_value=long_value, short_value=short_value,
            cash=float(latest.cash) if latest else None, day_change=day_change,
            day_change_pct=day_change / float(previous.equity) if previous and previous.equity else None,
            week_change=week_change,
            week_change_pct=week_change / float(week_base.equity) if week_base and week_base.equity else None,
            positions_count=len(positions), orders_today=orders_today, orders_rejected_today=rejected, alarms=[],
            spark_30d=spark if len(spark) >= 2 else None)


@app.get("/api/accounts/{account_id}/equity", response_model=list[EquityPoint])
def account_equity(account_id: str, range: Literal["1W", "1M", "3M", "YTD", "ALL"] = Query("1M"),
                   db: Session = Depends(get_db)):
    from quantagent.models import EquitySnapshot
    from .accounts import account_session_factory

    account = configured_account(account_id, db)
    with account_session_factory(account)() as source:
        if not inspect(source.get_bind()).has_table("equity_snapshots"):
            return []
        rows = source.scalars(select(EquitySnapshot).order_by(EquitySnapshot.snapshot_date)).all()
        if not rows:
            return []
        today = datetime.now(NEW_YORK).date()
        starts = {"1W": today - timedelta(days=7), "1M": today - timedelta(days=30),
                  "3M": today - timedelta(days=90), "YTD": today.replace(month=1, day=1), "ALL": None}
        start = starts[range]
        return [EquityPoint(date=row.snapshot_date.isoformat(), account=float(row.equity))
                for row in rows if start is None or row.snapshot_date >= start]


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


_web_dist = Path(os.environ.get("QUANTAGENT_WEB_DIST", Path(__file__).resolve().parents[2] / "web" / "dist"))
if _web_dist.is_dir():
    app.mount("/", SPAStaticFiles(directory=_web_dist, html=True), name="web")
