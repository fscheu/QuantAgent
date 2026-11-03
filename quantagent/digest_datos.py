"""Paper trading data queries used by the operational digest."""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

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


def reconstruir_posiciones(fills):
    """Consolidate ordered fills at average cost, including reductions and reversals.

    Used by the digest's realized P&L and the read-only web without a live broker.
    """
    positions, sales = {}, []
    for fill, order in fills:
        quantity = fill.quantity if order.side.value == "buy" else -fill.quantity
        price = fill.price
        position = positions.setdefault(order.symbol, {
            "quantity": Decimal(0), "average_entry_price": Decimal(0),
            "opened_at": fill.filled_at, "updated_at": fill.filled_at,
        })
        old = position["quantity"]
        cost = position["average_entry_price"]
        remaining = old + quantity
        if old * quantity < 0:
            closed = min(abs(old), abs(quantity))
            pnl = (price - cost) * closed * (1 if old > 0 else -1)
            sales.append({"symbol": order.symbol, "side": order.side.value,
                          "quantity": float(closed), "pnl": float(pnl - (fill.commission or 0)),
                          "filled_at": fill.filled_at})
        if old == 0 or old * remaining < 0:
            position["average_entry_price"] = price
            position["opened_at"] = fill.filled_at
        elif old * quantity > 0:
            position["average_entry_price"] = (cost * abs(old) + price * abs(quantity)) / abs(remaining)
        position["quantity"] = remaining
        position["updated_at"] = fill.filled_at
    return {symbol: position for symbol, position in positions.items() if position["quantity"]}, sales


def consultar_datos_cartera(session: Session, day: date, portfolio_name: str, broker) -> tuple[list[dict], list[dict], list[dict], list[dict], float]:
    """Read a dedicated portfolio DB; fills are executions, not realized P&L.

    Positions and pending statuses are the current local snapshot, not historical balances.
    """
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    positions = broker.get_positions()
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
    fills_historicos = (
        session.query(Fill, Order).join(Order, Fill.order_id == Order.id)
        .filter(Order.environment == Environment.PAPER, Fill.filled_at < end)
        .order_by(Fill.filled_at, Fill.id).all()
    )
    _, sales = reconstruir_posiciones(fills_historicos)
    realized = sum(sale["pnl"] for sale in sales)
    daily = [{key: value for key, value in sale.items() if key != "filled_at"}
             for sale in sales if sale["filled_at"] >= start]
    return (
        [{"name": portfolio_name, "positions": [
            {"symbol": p.symbol, "quantity": float(p.qty), "side": p.side,
             "price": float(p.current_price), "value": float(p.market_value),
             "unrealized_pnl": float(p.unrealized_pl)}
            for p in positions.values()]}],
        [{"symbol": o.symbol, "side": o.side.value, "quantity": f.quantity,
          "price": f.price, "commission": f.commission, "filled_at": f.filled_at}
         for f, o in fills],
        [{"symbol": o.symbol, "side": o.side.value, "status": o.status.value,
          "quantity": o.quantity, "filled_quantity": o.filled_quantity, "comment": o.comment}
         for o in pending], daily, realized,
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
