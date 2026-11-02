"""FastAPI application and read-only database dependency."""

import os
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.orm import Session, sessionmaker
from starlette.exceptions import HTTPException as StarletteHTTPException

from .feed import build_feed
from .schemas import (
    AccountRef,
    AccountSummaryActive,
    AccountSummaryNoData,
    BacktestRef,
    DividendResponse,
    EquityPoint,
    HomeStatus,
    OrderResponse,
    PositionResponse,
    SectorExposure,
)

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
    kind: Literal["holiday", "early_close", "rebalance", "macro"]
    starts_at: str | None = None
    date: str
    title: str
    subtitle: str | None = None
    source: Literal["alpaca", "strategy", "fred", "manual"]
    account_id: str | None = None


class FeedEventResponse(BaseModel):
    id: str
    type: str
    severity: str
    occurred_at: str
    account_id: str | None = None
    title: str
    subtitle: str | None = None
    link: str
    group_count: int


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


def _backtest_curve(run_id: int):
    import csv
    from pathlib import Path

    root = Path(os.environ.get("QUANTAGENT_BACKTEST_EQUITY_DIR", "local-data/backtest-equity"))
    path = root / f"{run_id}.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Backtest {run_id} exists but has no saved equity curve")
    with path.open(newline="", encoding="utf-8") as source:
        return [{"date": row["timestamp"], "equity": float(row["equity"])} for row in csv.DictReader(source)]


@app.get("/api/backtests", response_model=list[BacktestRef])
def backtests(hypothesis_id: str | None = None, q: str | None = None, db: Session = Depends(get_db)):
    from quantagent.models import BacktestRun

    rows = db.scalars(select(BacktestRun).order_by(BacktestRun.created_at.desc())).all()
    result = []
    for row in rows:
        config = row.config_snapshot or {}
        candidate_id = config.get("hypothesis_id")
        hypothesis_name = config.get("hypothesis_name")
        if hypothesis_id and str(candidate_id) != hypothesis_id:
            continue
        label = row.name or f"Backtest #{row.id}"
        if q and q.casefold() not in f"{label} {hypothesis_name or ''}".casefold():
            continue
        try:
            curve = _backtest_curve(row.id)
        except HTTPException:
            curve = []
        total_return = (curve[-1]["equity"] / curve[0]["equity"] - 1) * 100 if len(curve) > 1 and curve[0]["equity"] else 0.0
        result.append(BacktestRef(id=row.id, label=label, hypothesis_name=hypothesis_name,
                                  start=row.start_date.date().isoformat(), end=row.end_date.date().isoformat(),
                                  total_return=total_return))
    return result


@app.get("/api/performance/backtest/{backtest_id}")
def backtest_performance(backtest_id: int, range_: Literal["ytd", "1y", "3y", "5y", "all"] = Query("all", alias="range"), align_with: str | None = Query(None), db: Session = Depends(get_db)):
    from quantagent.data.snapshot import SnapshotError, read_snapshot
    from quantagent.models import BacktestRun
    from quantagent.performance import build_report

    run = db.get(BacktestRun, backtest_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    equity = _backtest_curve(backtest_id)
    if not equity:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} has no saved equity curve")
    align = None
    if align_with:
        other_id = int(align_with.split(":")[-1])
        if db.get(BacktestRun, other_id) is None:
            raise HTTPException(status_code=404, detail=f"Comparison backtest {other_id} not found")
        align = _backtest_curve(other_id)
    snapshot_name = os.environ.get("QUANTAGENT_WEB_SNAPSHOT")
    try:
        spy = read_snapshot(snapshot_name, "SPY")["close"] if snapshot_name else []
    except (SnapshotError, OSError, ValueError, KeyError):
        spy = []
    config = run.config_snapshot or {}
    subject = {"type": "backtest", "id": str(run.id), "name": run.name or f"Backtest #{run.id}",
               "env": None, "run_label": f"#{run.id}", "account_id": None,
               "hypothesis_id": config.get("hypothesis_id"), "backtest_id": str(run.id),
               "out_of_sample_start": None, "costs_bps": None, "strategies": None}
    return build_report(equity, spy, subject=subject, range=range_, align_with=align)


@app.get("/api/performance/portfolio/{account_id}")
def portfolio_performance(account_id: str, range_: Literal["ytd", "1y", "3y", "5y", "all"] = Query("all", alias="range"), db: Session = Depends(get_db)):
    from sqlalchemy import inspect

    from quantagent.models import CashFlow, EquitySnapshot
    from quantagent.performance import build_report

    from .accounts import account_session_factory

    account = configured_account(account_id, db)
    with account_session_factory(account)() as source:
        if not inspect(source.get_bind()).has_table("equity_snapshots"):
            raise HTTPException(status_code=404, detail="Account has no equity snapshots")
        equity = [{"date": row.snapshot_date, "equity": row.equity} for row in
                  source.scalars(select(EquitySnapshot).order_by(EquitySnapshot.snapshot_date)).all()]
        registered = inspect(source.get_bind()).has_table("cash_flows")
        flows = ([{"date": row.flow_date, "amount": row.amount, "reconciled": row.reconciled}
                  for row in source.scalars(select(CashFlow).order_by(CashFlow.flow_date)).all()]
                 if registered else None)
    subject = {"type": "portfolio", "id": account.id, "name": account.name, "env": account.env,
               "account_id": account.id, "strategies": None}
    report = build_report(equity, [], subject=subject, range=range_, flows=flows)
    report["flows"]["registered"] = registered
    return report


@app.get("/api/performance/run/{account_id}")
def run_performance(account_id: str, range_: Literal["ytd", "1y", "3y", "5y", "all"] = Query("all", alias="range"), db: Session = Depends(get_db)):
    report = portfolio_performance(account_id, range_, db)
    account = configured_account(account_id, db)
    report["subject"].update(type="run", name=account.strategy or account.name,
                             run_label=account.id, backtest_id=None)
    return report


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


@app.get("/api/feed", response_model=list[FeedEventResponse])
def feed(since: datetime | None = Query(None), limit: int = Query(50, ge=1, le=200),
         db: Session = Depends(get_db)):
    return [FeedEventResponse(**row) for row in build_feed(db, since=since, limit=limit, utc_z=utc_z)]


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
    from .alarms import evaluate_account_alarms

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
            positions_count=len(positions), orders_today=orders_today, orders_rejected_today=rejected,
            alarms=[alarm.model_dump() for alarm in evaluate_account_alarms(account, db, datetime.now(ZoneInfo("UTC")))],
            spark_30d=spark if len(spark) >= 2 else None)


@app.get("/api/home/status", response_model=HomeStatus)
def home_status(db: Session = Depends(get_db)):
    from quantagent.models import Account, Position

    from .accounts import account_session_factory
    from .alarms import evaluate_account_alarms

    alarms = []
    for account in db.scalars(select(Account).where(Account.status == "active").order_by(Account.id)).all():
        with account_session_factory(account)() as source:
            if not source.scalars(select(Position).limit(1)).first():
                continue
        alarms.extend(evaluate_account_alarms(account, db, datetime.now(ZoneInfo("UTC"))))
    rank = {"critical": 0, "warning": 1, "info": 2, "neutral": 3}
    alarms.sort(key=lambda alarm: (rank[alarm.severity], alarm.since))
    if not alarms:
        return HomeStatus(level="ok", headline="Todo en orden", message="Ninguna cuenta tiene alarmas.", link=None)
    first = alarms[0]
    count = sum(a.severity == first.severity for a in alarms)
    if first.severity == "critical":
        headline = f"Atención: {count} alarma crítica" + ("s" if count != 1 else "")
    else:
        headline = f"Casi todo bien: {count} advertencia" + ("s" if count != 1 else "")
    return HomeStatus(level=first.severity, headline=headline, message=first.message, link=f"/cuenta/{first.account_id}")


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
        selected = [row for row in rows if start is None or row.snapshot_date >= start]
        from .account_extras import equity_extras
        benchmark, shadow = equity_extras(account, selected)
        return [EquityPoint(date=row.snapshot_date.isoformat(), account=float(row.equity),
                            benchmark=benchmark[row.snapshot_date.isoformat()],
                            shadow=shadow[row.snapshot_date.isoformat()]) for row in selected]


@app.get("/api/accounts/{account_id}/dividends", response_model=list[DividendResponse])
def account_dividends(account_id: str, db: Session = Depends(get_db)):
    from quantagent.models import Event

    configured_account(account_id, db)
    rows = db.scalars(select(Event).where(Event.account_id == account_id,
        Event.type == "dividend_received").order_by(Event.occurred_at.desc(), Event.id.desc())).all()
    return [DividendResponse(date=utc_z(row.occurred_at), symbols=row.payload.get("symbols", []),
        net=float(row.payload.get("net", 0)), withheld=float(row.payload.get("withheld", 0))) for row in rows]


@app.get("/api/accounts/{account_id}/sector-exposure", response_model=list[SectorExposure])
def account_sector_exposure(account_id: str, db: Session = Depends(get_db)):
    from .account_extras import sector_exposure
    from .accounts import account_session_factory

    account = configured_account(account_id, db)
    with account_session_factory(account)() as source:
        return sector_exposure(account, source, datetime.now(NEW_YORK).date())


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
