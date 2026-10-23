from quantagent.models import OrderSide
from quantagent.portfolio.manager import PortfolioManager
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.rebalanceador import ordenes_rebalanceo, rebalancear


def test_rebalance_existing_positions_sells_before_buying(db_session):
    portfolio = PortfolioManager(2000, db=db_session)
    portfolio.positions = {"A": {"qty": 10, "avg_cost": 100, "current_price": 100, "pnl": 0, "pnl_pct": 0},
                           "B": {"qty": 70, "avg_cost": 100, "current_price": 100, "pnl": 0, "pnl_pct": 0}}
    manager = OrderManager(PositionSizer(), RiskManager(portfolio, db=db_session), PaperBroker(slippage_pct=0), portfolio, db_session, cash_reserve_pct=.05)
    orders = rebalancear({"A": .45, "B": .10, "C": .40}, {"A": 100, "B": 100, "C": 100}, manager)
    assert [o.side for o in orders if o] == [OrderSide.SELL, OrderSide.BUY, OrderSide.BUY]
    assert {s: p["qty"] for s, p in portfolio.positions.items()} == {"A": 45, "B": 10, "C": 40}


def test_already_at_weights_creates_no_orders():
    adjustments = ordenes_rebalanceo({"A": .5, "B": .5}, {"A": {"qty": 5}, "B": {"qty": 5}}, {"A": 100, "B": 100}, 1000)
    assert adjustments == []
