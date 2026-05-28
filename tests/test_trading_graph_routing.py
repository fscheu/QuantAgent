from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from quantagent.backtesting.backtest import Backtest
from quantagent.llm.roles import ProviderRoleConfig
from quantagent.llm.routing import ProviderRoutingPolicy
from quantagent.models import BacktestRun, StrategyConfig
from quantagent.trading_graph import TradingGraph


class DummySetGraph:
    def __init__(self, *args, **kwargs):
        pass

    def set_graph(self, **kwargs):
        return {"graph": "ok"}


def _sample_policy() -> ProviderRoutingPolicy:
    return ProviderRoutingPolicy(
        deep_reasoning=ProviderRoleConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            temperature=0.05,
        ),
        lite=ProviderRoleConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.15,
        ),
    )


def test_trading_graph_default_init(monkeypatch):
    created = []

    def fake_create_llm(self, provider, model, temperature):
        created.append((provider, model, temperature))
        return {"provider": provider, "model": model, "temperature": temperature}

    monkeypatch.setattr("quantagent.trading_graph.TechnicalTools", lambda: object())
    monkeypatch.setattr("quantagent.trading_graph.SetGraph", DummySetGraph)
    monkeypatch.setattr(TradingGraph, "_create_llm", fake_create_llm)

    graph = TradingGraph()

    assert len(created) == 2
    assert graph._routing_policy is not None
    assert graph._routing_policy.deep_reasoning is not None
    assert graph._routing_policy.lite is not None


def test_routing_policy_db_persistence(db_session):
    policy = _sample_policy()
    db_session.add(
        StrategyConfig(
            name="cost-efficient-default",
            kind="provider_routing",
            json_config=policy.to_dict(),
        )
    )
    db_session.commit()

    loaded = (
        db_session.query(StrategyConfig)
        .filter_by(name="cost-efficient-default", kind="provider_routing")
        .one()
    )
    restored = ProviderRoutingPolicy.from_dict(loaded.json_config)

    assert restored.resolve("deep_reasoning") == policy.resolve("deep_reasoning")
    assert restored.resolve("lite") == policy.resolve("lite")
    assert restored.resolve("image") == policy.resolve("image")


def test_backtest_metadata_includes_roles(db_session, monkeypatch):
    policy = _sample_policy()
    components = SimpleNamespace(
        graph=object(),
        portfolio_manager=SimpleNamespace(cash=100000.0, get_total_value=lambda: 100000.0),
        order_manager=object(),
    )

    monkeypatch.setattr(
        "quantagent.backtesting.backtest.StrategyAssembler.build_components",
        lambda resolved, db_session: components,
    )

    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        assets=["BTC"],
        timeframe="1h",
        initial_capital=100000.0,
        config={"routing_policy": policy.to_dict()},
        db_session=db_session,
        strategy=MagicMock(required_history_bars=30),
    )

    backtest._create_backtest_run(name="Routing metadata")

    run = (
        db_session.query(BacktestRun)
        .filter(BacktestRun.id == backtest.backtest_run_id)
        .one()
    )

    provider_roles_used = run.config_snapshot["provider_roles_used"]
    assert provider_roles_used["deep_reasoning"]["provider"] == "anthropic"
    assert provider_roles_used["deep_reasoning"]["model"] == "claude-haiku-4-5-20251001"
    assert provider_roles_used["lite"]["provider"] == "openai"
    assert provider_roles_used["image"]["role"] == "image"
    assert provider_roles_used["image"]["resolved_from"] == "deep_reasoning"
