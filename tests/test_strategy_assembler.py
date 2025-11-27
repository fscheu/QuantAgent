import os
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.strategy.assembler import StrategyAssembler
from quantagent.models import Base, Environment, Signal, TradeSignal, Order, OrderStatus


def make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return Session()


def test_build_components_from_snapshot():
    db = make_session()
    # Ensure API key exists in env for TradingGraph init
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    snapshot = {
        "initial_cash": 50000.0,
        "base_position_pct": 0.05,
        "max_daily_loss_pct": 0.05,
        "max_position_pct": 0.10,
        "slippage_pct": 0.01,
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "temperature": 0.1,
        "use_checkpointing": False,
        "universe": ["BTC", "ETH"],
    }

    resolved = StrategyAssembler.from_snapshot(snapshot, environment=Environment.BACKTEST)
    components = StrategyAssembler.build_components(resolved, db_session=db)

    assert components.portfolio_manager is not None
    assert components.position_sizer is not None
    assert components.risk_manager is not None
    assert components.broker is not None
    assert components.order_manager is not None
    assert components.graph is not None


def test_order_creation_persists_environment_and_trigger_signal_id():
    db = make_session()
    # Ensure API key exists in env for TradingGraph init
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    # Prepare resolved config and components
    resolved = StrategyAssembler.from_profiles(
        portfolio_profile={
            "universe": ["BTC"],
            "initial_cash": 100000.0,
            "base_position_pct": 0.05,
            "slippage_pct": 0.01,
        },
        risk_profile={
            "max_daily_loss_pct": 0.05,
            "max_position_pct": 0.10,
        },
        model_profile={
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "use_checkpointing": False,
        },
        environment=Environment.BACKTEST,
    )
    components = StrategyAssembler.build_components(resolved, db_session=db)

    # Seed a Signal that will trigger the order (provenance)
    sig = Signal(
        symbol="BTC",
        signal=TradeSignal.LONG,
        confidence=0.9,
        timeframe="1h",
        environment=Environment.BACKTEST,
        generated_at=datetime.utcnow(),
        analysis_summary="Test analysis",
    )
    db.add(sig)
    db.commit()

    # Execute decision and persist Order with provenance
    filled = components.order_manager.execute_decision(
        symbol="BTC",
        decision="LONG",
        confidence=1.0,
        current_price=10000.0,
        environment=Environment.BACKTEST,
        trigger_signal_id=sig.id,
    )

    assert filled is not None

    # Validate Order persisted with environment and trigger_signal_id
    saved_order = db.query(Order).filter(Order.symbol == "BTC").first()
    assert saved_order is not None
    assert saved_order.environment == Environment.BACKTEST
    assert saved_order.trigger_signal_id == sig.id
    assert saved_order.status == OrderStatus.FILLED

    # Validate reverse link also stored (Signal.order_id)
    saved_signal = db.query(Signal).filter(Signal.id == sig.id).first()
    assert saved_signal.order_id == saved_order.id
