"""Preparación de stops y vencimiento de entradas antes de abrir el mercado."""

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from quantagent.models import ActivePosition, Environment, Order, OrderSide, OrderType
from quantagent.trading.cartera_paper import sesiones_cerradas_nyse


def _ordenes_abiertas(client):
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        filtro = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    except ImportError:
        filtro = SimpleNamespace(status="open")
    return client.get_orders(filter=filtro)


def _campo(obj, nombre):
    return getattr(getattr(obj, nombre, None), "value", getattr(obj, nombre, None))


def correr_pre_apertura(session, broker, ahora=None, max_sesiones_entrada=3, aviso_day_trades=2):
    ahora = ahora or datetime.utcnow()
    posiciones = broker.get_positions()
    locales = {p.symbol: p for p in session.query(ActivePosition).filter(
        ActivePosition.is_active.is_(True), ActivePosition.quantity > 0,
        ActivePosition.environment == Environment.PAPER, ActivePosition.side == OrderSide.BUY,
    ).all()}
    ordenes = list(_ordenes_abiertas(broker.client))
    stops = [o for o in ordenes if _campo(o, "type") == "stop" and _campo(o, "side") == "sell"]
    reenviados = []
    for symbol, posicion in posiciones.items():
        qty = Decimal(str(posicion.qty))
        if _campo(posicion, "side") != "long" or qty == qty.to_integral_value():
            continue
        if any(o.symbol == symbol for o in stops):
            continue
        local = locales.get(symbol)
        if local is None:
            continue
        orden = Order(symbol=symbol, side=OrderSide.SELL, order_type=OrderType.STOP,
                      quantity=qty, stop_price=Decimal(str(local.stop_loss)), environment=Environment.PAPER)
        broker.submit_order(orden)
        reenviados.append(symbol)

    sesiones = sesiones_cerradas_nyse(ahora, n_sesiones=max_sesiones_entrada + 1)
    primer_dia_vigente = sesiones[-max_sesiones_entrada].date()
    vencidas = []
    for orden in ordenes:
        if (_campo(orden, "type") == "stop" and _campo(orden, "side") == "buy"
                and orden.created_at.date() < primer_dia_vigente):
            order_id = getattr(orden, "id", None)
            if order_id is not None and broker.cancel_order(str(order_id)):
                vencidas.append(str(order_id))

    ordenes = list(_ordenes_abiertas(broker.client))
    stops = [o for o in ordenes if _campo(o, "type") == "stop" and _campo(o, "side") == "sell"]
    sin_stop = []
    for symbol, posicion in posiciones.items():
        qty = Decimal(str(posicion.qty))
        cubiertas = [Decimal(str(o.qty)) for o in stops if o.symbol == symbol]
        if not cubiertas or max(cubiertas) < qty:
            sin_stop.append(symbol)

    avisos = []
    day_trades = int(broker.client.get_account().daytrade_count)
    if day_trades >= aviso_day_trades:
        avisos.append(f"Alerta PDT: day_trade_count={day_trades}; umbral={aviso_day_trades}.")
    return {"reenviados": reenviados, "entradas_vencidas": vencidas,
            "sin_stop": sorted(sin_stop), "avisos": avisos}
