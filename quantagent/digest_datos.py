"""Paper trading data queries used by the operational digest."""

from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy.orm import Session

from quantagent.models import ActivePosition, Environment, Fill, Order, OrderStatus, Position, SchedulerHeartbeat, Trade


def crear_broker_alpaca_paper():
    """Build the paper broker used by operational trading commands."""
    from quantagent.strategy.assembler import StrategyAssembler

    resolved = StrategyAssembler.from_profiles(
        overrides={"broker_type": "alpaca_paper"}, environment=Environment.PAPER
    )
    return StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"](resolved)


def consultar_valuacion_actual(broker, portfolios: list[dict]) -> dict:
    """Read AlpacaBroker's current snapshot, independently of the digest's --date.

    Return enriched copies; failures must be surfaced by the caller, never replaced
    with local entry prices or stored P&L.
    """
    consulted_at = datetime.now(timezone.utc)
    try:
        account = broker.get_account()
        valuations = broker.get_positions()
    except Exception as exc:
        raise RuntimeError(f"Valuación actual: error al consultar Alpaca ({type(exc).__name__}).") from exc
    valued_portfolios = []
    for portfolio in portfolios:
        positions = []
        for position in portfolio["positions"]:
            symbol = position["symbol"]
            if symbol not in valuations:
                raise ValueError(f"Valuación actual: Alpaca sin valuación para {symbol}.")
            remote = valuations[symbol]
            positions.append({**position, "price": float(remote.current_price),
                              "value": float(remote.market_value),
                              "unrealized_pnl": float(remote.unrealized_pl)})
        valued_portfolios.append({**portfolio, "positions": positions})
    return {"equity": float(account["equity"]), "cash": float(account["cash"]),
            "portfolios": valued_portfolios, "source": "Alpaca", "consulted_at": consulted_at,
            "notice": "Foto actual de Alpaca; no es saldo histórico de --date."}


def consultar_datos_cartera(session: Session, day: date, portfolio_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Read a dedicated portfolio DB; fills are executions, not realized P&L.

    Positions and pending statuses are the current local snapshot, not historical balances.
    """
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    positions = session.query(Position).order_by(Position.symbol).all()
    fills = (
        session.query(Fill, Order).join(Order, Fill.order_id == Order.id)
        .filter(Order.environment == Environment.PAPER, Fill.filled_at >= start, Fill.filled_at < end)
        .order_by(Fill.filled_at, Fill.id).all()
    )
    pending = (
        session.query(Order)
        .filter(Order.environment == Environment.PAPER,
                Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]))
        .order_by(Order.created_at, Order.id).all()
    )
    return (
        [{"name": portfolio_name, "positions": [
            {"symbol": p.symbol, "quantity": p.quantity, "price": p.current_price,
             "value": p.quantity * p.current_price, "unrealized_pnl": p.unrealized_pnl}
            for p in positions]}],
        [{"symbol": o.symbol, "side": o.side.value, "quantity": f.quantity,
          "price": f.price, "commission": f.commission, "filled_at": f.filled_at}
         for f, o in fills],
        [{"symbol": o.symbol, "side": o.side.value, "status": o.status.value,
          "quantity": o.quantity, "filled_quantity": o.filled_quantity, "comment": o.comment}
         for o in pending],
    )


def consultar_datos_paper(session: Session, day: date, portfolio_name: str) -> tuple[list[dict], list[dict], list[dict], float, list[dict]]:
    """Return paper digest data; Position lacks environment, so symbols come from paper ActivePosition."""
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    heartbeat_start = start - timedelta(days=6)
    paper_symbols = {
        symbol for (symbol,) in session.query(ActivePosition.symbol)
        .filter(ActivePosition.environment == Environment.PAPER,
                ActivePosition.is_active.is_(True), ActivePosition.decision_timestamp < end)
        .all()
    }
    positions = (
        session.query(Position).filter(Position.symbol.in_(paper_symbols))
        .order_by(Position.symbol).all()
    )
    trades = (
        session.query(Trade)
        .filter(Trade.environment == Environment.PAPER, Trade.closed_at >= start, Trade.closed_at < end)
        .order_by(Trade.closed_at)
        .all()
    )
    orders = (
        session.query(Order)
        .filter(Order.environment == Environment.PAPER, Order.created_at >= start, Order.created_at < end)
        .order_by(Order.created_at)
        .all()
    )
    realized_pnl = (
        session.query(Trade.pnl)
        .filter(Trade.environment == Environment.PAPER, Trade.closed_at < end)
        .all()
    )
    heartbeats = (
        session.query(SchedulerHeartbeat)
        .filter(SchedulerHeartbeat.environment == Environment.PAPER,
                SchedulerHeartbeat.timestamp >= heartbeat_start, SchedulerHeartbeat.timestamp < end)
        .order_by(SchedulerHeartbeat.timestamp)
        .all()
    )
    return (
        [{"name": portfolio_name, "positions": [
            {"symbol": p.symbol, "quantity": p.quantity, "price": p.current_price,
             "value": p.quantity * p.current_price,
             "unrealized_pnl": p.unrealized_pnl} for p in positions
        ]}],
        [{"symbol": t.symbol, "side": t.side.value, "quantity": t.quantity, "pnl": t.pnl} for t in trades],
        [{"timestamp": h.timestamp, "status": h.status, "error_message": h.error_message} for h in heartbeats],
        sum((pnl or 0) for (pnl,) in realized_pnl),
        [{"symbol": o.symbol, "side": o.side.value, "status": o.status.value,
          "quantity": o.quantity, "filled_quantity": o.filled_quantity, "comment": o.comment}
         for o in orders],
    )
