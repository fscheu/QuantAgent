"""Paper portfolio target-weight rebalancing."""

from math import floor

from quantagent.models import OrderSide


def ordenes_rebalanceo(pesos, posiciones, precios, valor):
    """Return integer share adjustments, selling before buying."""
    symbols = set(pesos) | set(posiciones)
    ajustes = []
    for symbol in symbols:
        price = precios[symbol]
        held = int(posiciones.get(symbol, {}).get("qty", 0))
        target = floor(max(0, pesos.get(symbol, 0) * valor / price))
        delta = target - held
        if delta:
            ajustes.append((symbol, OrderSide.SELL if delta < 0 else OrderSide.BUY, abs(delta), price))
    return sorted(ajustes, key=lambda row: row[1] == OrderSide.BUY)


def rebalancear(pesos, precios, order_manager):
    portfolio = order_manager.portfolio
    value = portfolio.get_total_value()
    orders = ordenes_rebalanceo(pesos, portfolio.positions, precios, value)
    return [order_manager.execute_rebalance_order(*order) for order in orders]
