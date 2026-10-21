from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.database import Base
from quantagent.models import Environment, OrderSide, Trade
from quantagent.portfolio.manager import PortfolioManager
from quantagent.settings import SchedulerSettings
from quantagent.trading.order_manager import OrderManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.scheduler import TradingScheduler


@pytest.mark.parametrize("side,qty", [(OrderSide.BUY, 1.0), (OrderSide.SELL, -1.0)])
@pytest.mark.parametrize("with_trade_id", [True, False])
def test_paper_stop_exit_leaves_portfolio_flat(side, qty, with_trade_id):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        trade = Trade(
            symbol="BTC", entry_price=100, quantity=abs(qty), side=side,
            environment=Environment.PAPER,
        )
        session.add(trade)
        session.commit()
        portfolio = PortfolioManager(100_000, Environment.PAPER, db=session)
        portfolio.positions["BTC"] = {
            "qty": qty, "avg_cost": 100.0, "current_price": 100.0,
            "pnl": 0.0, "pnl_pct": 0.0,
        }
        position = PositionMonitor(session).open_position(
            symbol="BTC", side=side, entry_price=100,
            stop_loss=98, take_profit=110, quantity=Decimal("1"),
            exit_policy="sl_tp_only", trade_id=trade.id if with_trade_id else None,
        )
        position.environment = Environment.PAPER
        session.commit()
        order_manager = OrderManager(
            PositionSizer(base_position_pct=0.05), RiskManager(portfolio, db=session),
            PaperBroker(slippage_pct=0), portfolio, session,
        )
        scheduler = TradingScheduler(
            trading_graph=None, order_manager=order_manager, data_provider=None,
            db_session=session,
            scheduler_settings=SchedulerSettings(
                enabled=False, interval_hours=1, assets=["BTC"],
                environment="paper", timeframe="1h", lookback_hours=12,
            ),
        )

        scheduler._execute_position_exit(position, "stop_loss", 97, "BTC")

        session.refresh(position)
        observed = (portfolio.get_position("BTC"), position.is_active)
        print(f"portfolio_position={observed[0]}, active_position={observed[1]}")
        assert observed[0]["qty"] == 0
        assert observed[1] is False
        if with_trade_id:
            assert trade.exit_signal == "stop_loss"
