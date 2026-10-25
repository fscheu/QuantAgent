"""Paper trading data queries used by the operational digest."""

from datetime import date, datetime, time, timedelta

from sqlalchemy.orm import Session

from quantagent.models import ActivePosition, Environment, Position, SchedulerHeartbeat, Trade


def consultar_datos_paper(session: Session, day: date, portfolio_name: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Return digest portfolios, closed paper trades, and paper heartbeats for a day."""
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
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
    heartbeats = (
        session.query(SchedulerHeartbeat)
        .filter(SchedulerHeartbeat.environment == Environment.PAPER,
                SchedulerHeartbeat.timestamp >= start, SchedulerHeartbeat.timestamp < end)
        .order_by(SchedulerHeartbeat.timestamp)
        .all()
    )
    return (
        [{"name": portfolio_name, "positions": [
            {"symbol": p.symbol, "quantity": p.quantity, "price": p.current_price,
             "unrealized_pnl": p.unrealized_pnl} for p in positions
        ]}],
        [{"symbol": t.symbol, "side": t.side.value, "quantity": t.quantity, "pnl": t.pnl} for t in trades],
        [{"timestamp": h.timestamp, "status": h.status, "error_message": h.error_message} for h in heartbeats],
    )
