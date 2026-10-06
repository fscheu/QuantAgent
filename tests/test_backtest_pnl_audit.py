from datetime import datetime
from decimal import Decimal
from quantagent.models import Environment, Order, OrderSide, OrderStatus, OrderType
from quantagent.portfolio.manager import PortfolioManager
from quantagent.backtesting.backtest import Backtest


def test_hand_calculated_trades_pnl_and_metrics(db_session):
    """Audit PnL calculation with 3 hand-calculated trades: 1 long winner, 1 long loser, 1 short."""
    start = datetime(2026, 1, 1, 10, 0)
    end = datetime(2026, 1, 2, 18, 0)
    initial_capital = 10000.0
    portfolio = PortfolioManager(db=db_session, initial_cash=initial_capital, environment=Environment.BACKTEST)

    def _exec(side, qty, price):
        order = Order(
            symbol="SPY",
            side=side,
            quantity=Decimal(str(qty)),
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            environment=Environment.BACKTEST,
        )
        db_session.add(order)
        db_session.commit()
        return portfolio.execute_trade(order, fill_price=price, timestamp=start)

    # 1. Long winner: buy 10 @ 100, sell 10 @ 110 -> pnl = (110 - 100) * 10 = +100.0
    _exec(OrderSide.BUY, 10, 100.0)
    t1 = _exec(OrderSide.SELL, 10, 110.0)
    assert t1.pnl == Decimal("100.0")

    # 2. Long loser: buy 10 @ 100, sell 10 @ 95 -> pnl = (95 - 100) * 10 = -50.0
    _exec(OrderSide.BUY, 10, 100.0)
    t2 = _exec(OrderSide.SELL, 10, 95.0)
    assert t2.pnl == Decimal("-50.0")

    # 3. Short winner: sell 5 @ 100, buy 5 @ 90 -> pnl = (100 - 90) * 5 = +50.0
    _exec(OrderSide.SELL, 5, 100.0)
    t3 = _exec(OrderSide.BUY, 5, 90.0)
    assert t3.pnl == Decimal("50.0")

    # Engine metrics verification: total_pnl = 100 - 50 + 50 = 100.0; return = 100/10000 = 1.0%
    bt = Backtest(
        start_date=start,
        end_date=end,
        assets=["SPY"],
        initial_capital=initial_capital,
        config={},
        db_session=db_session,
    )
    metrics = bt._calculate_metrics()
    assert metrics.total_pnl == 100.0
    assert metrics.total_return_pct == 1.0
    assert portfolio.cash - initial_capital == 100.0
