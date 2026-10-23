import os
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import Base, Environment, Order, OrderStatus, Signal, TradeSignal
from quantagent.strategy.assembler import StrategyAssembler


def make_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return Session()


def test_broker_selector_uses_registered_constructor():
    db = make_session()
    marker = object()
    original = StrategyAssembler.BROKER_CONSTRUCTORS.get("alpaca_paper")
    StrategyAssembler.register_broker("alpaca_paper", lambda config: marker)
    try:
        resolved = StrategyAssembler.from_profiles(
            portfolio_profile={"broker_type": "alpaca_paper"}
        )
        components = StrategyAssembler.build_components(
            resolved, db_session=db, build_graph=False
        )
        assert components.broker is marker
    finally:
        StrategyAssembler.BROKER_CONSTRUCTORS["alpaca_paper"] = original


def test_resolved_risk_per_trade_reaches_position_sizer():
    db = make_session()
    resolved = StrategyAssembler.from_snapshot({"riesgo_por_trade_pct": 0.003})
    components = StrategyAssembler.build_components(resolved, db_session=db, build_graph=False)
    assert resolved.riesgo_por_trade_pct == 0.003
    assert components.position_sizer.riesgo_por_trade_pct == 0.003


def test_alpaca_paper_builds_configured_broker_and_blocks_live(monkeypatch):
    import sys
    from types import ModuleType

    db = make_session()
    calls = {}

    class TradingClient:
        def __init__(self, key, secret, paper):
            calls.update(key=key, secret=secret, paper=paper)

    alpaca = ModuleType("alpaca")
    trading = ModuleType("alpaca.trading")
    client_module = ModuleType("alpaca.trading.client")
    client_module.TradingClient = TradingClient
    monkeypatch.setitem(sys.modules, "alpaca", alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", client_module)
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    resolved = StrategyAssembler.from_profiles(
        overrides={"broker_type": "alpaca_paper", "alpaca_base_url": "https://api.alpaca.markets"}
    )
    with pytest.raises(ValueError, match="live trading is not supported"):
        StrategyAssembler.build_components(resolved, db_session=db, build_graph=False)
    paper = StrategyAssembler.from_profiles(
        overrides={"broker_type": "alpaca_paper"}, environment=Environment.PAPER
    )
    components = StrategyAssembler.build_components(paper, db_session=db, build_graph=False)
    assert components.order_manager.cash_reserve_pct == 0.05
    from quantagent.trading.alpaca_broker import AlpacaBroker
    assert isinstance(components.broker, AlpacaBroker)
    assert calls == {"key": "paper-key", "secret": "paper-secret", "paper": True}


@pytest.mark.api
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

    resolved = StrategyAssembler.from_snapshot(
        snapshot, environment=Environment.BACKTEST
    )
    components = StrategyAssembler.build_components(resolved, db_session=db)

    assert components.portfolio_manager is not None
    assert components.position_sizer is not None
    assert components.risk_manager is not None
    assert components.broker is not None
    assert components.order_manager is not None
    assert components.order_manager.cash_reserve_pct == 0.0
    assert components.graph is not None


@pytest.mark.api
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
