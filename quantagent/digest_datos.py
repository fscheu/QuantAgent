"""Paper trading data queries used by the operational digest."""

from datetime import date, datetime, time, timedelta

from sqlalchemy.orm import Session

from quantagent.models import ActivePosition, Environment, Order, Position, SchedulerHeartbeat, Trade


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
