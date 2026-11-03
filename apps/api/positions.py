"""Consolidated positions from persisted fills and market prices; no broker calls."""

from sqlalchemy import inspect, select

from quantagent.digest_datos import reconstruir_posiciones
from quantagent.models import ActivePosition, Fill, MarketData, Order, OrderStatus

from .schemas import PositionResponse


def read_positions(db, environment):
    tables = inspect(db.get_bind())
    fills = db.execute(select(Fill, Order).join(Order, Fill.order_id == Order.id)
        .where(Order.environment == environment).order_by(Fill.filled_at, Fill.id)).all() if tables.has_table("fills") else []
    positions, _ = reconstruir_posiciones(fills)
    pending = db.scalars(select(Order).where(Order.environment == environment,
        Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]),
        Order.quantity > Order.filled_quantity).order_by(Order.created_at, Order.id)).all() if tables.has_table("orders") else []
    executed_symbols = set(positions)
    pending_symbols = {order.symbol for order in pending}
    # Pending-only symbols have no executed holding and must not enter shadow weights.
    for order in pending:
        if order.symbol not in positions:
            positions[order.symbol] = {"quantity": (order.quantity - order.filled_quantity) * (1 if order.side.value == "buy" else -1),
                "average_entry_price": order.price, "opened_at": order.created_at, "updated_at": order.updated_at}
    prices = {}
    if tables.has_table("market_data"):
        rows = db.execute(select(MarketData.symbol, MarketData.close)
            .where(MarketData.symbol.in_(positions)).order_by(MarketData.timestamp, MarketData.id)).all()
        prices = {symbol: float(price) for symbol, price in rows}
    stops = {}
    if tables.has_table("active_positions"):
        # Select only columns available in the legacy account schema.
        stops = {(symbol, side.value): float(stop) if stop is not None else None
                 for symbol, side, stop in db.execute(select(ActivePosition.symbol, ActivePosition.side, ActivePosition.stop_loss)
                     .where(ActivePosition.environment == environment, ActivePosition.is_active.is_(True))).all()}
    result = []
    for symbol, position in sorted(positions.items()):
        quantity = float(position["quantity"])
        side = "long" if quantity > 0 else "short"
        qty = abs(quantity)
        entry = float(position["average_entry_price"]) if position["average_entry_price"] is not None else None
        price = prices.get(symbol)
        # A fully closed symbol with a new unfilled order is pending-only too.
        executed = symbol in executed_symbols
        pnl = (price - entry) * quantity if executed and price is not None and entry is not None else None
        stop = stops.get((symbol, "buy" if side == "long" else "sell")) if executed else None
        result.append(PositionResponse(symbol=symbol, side=side,
            status="pending" if symbol in pending_symbols else "executed", qty=qty,
            avg_entry_price=entry, current_price=price,
            market_value=qty * price if executed and price is not None else None,
            unrealized_pl=pnl, unrealized_pl_pct=pnl / (qty * entry) if pnl is not None and entry else None,
            stop_price=stop, stop_distance=(price - stop if side == "long" else stop - price) if stop is not None and price is not None else None,
            opened_at=position["opened_at"].isoformat() + "Z"))
    return result
