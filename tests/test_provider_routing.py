from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quantagent.llm.registry import get_capability, supported_providers
from quantagent.llm.roles import ProviderRoleConfig
from quantagent.llm.routing import (
    ROLE_DEEP_REASONING,
    ROLE_IMAGE,
    ROLE_LITE,
    ProviderRoleNotConfiguredError,
    ProviderRoutingPolicy,
)
from quantagent.models import Environment
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.trading_graph import TradingGraph


class DummySetGraph:
    def __init__(self, *args, **kwargs):
        pass

    def set_graph(self, **kwargs):
        return {"graph": "ok"}


def test_supported_providers_are_stable():
    assert supported_providers() == ["anthropic", "azure", "openai", "qwen"]
    assert get_capability("openai").default_lite_model == "gpt-4o-mini"
    assert get_capability("anthropic").supports_vision is True


def test_get_capability_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_capability("made-up")


def test_provider_role_config_roundtrip_and_validation():
    role = ProviderRoleConfig(
        provider="qwen",
        model_name="qwen3-max",
        temperature=0.25,
        timeout_seconds=45,
        max_retries=2,
        capability_tags=["reasoning"],
    )

    payload = role.to_dict()

    assert ProviderRoleConfig.from_dict(payload) == role
    role.validate_against_registry()


def test_routing_policy_resolves_fallbacks_and_roundtrips():
    deep = ProviderRoleConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.1,
        capability_tags=["reasoning"],
    )
    lite = ProviderRoleConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.2,
        capability_tags=["cheap"],
    )
    policy = ProviderRoutingPolicy(deep_reasoning=deep, lite=lite)

    assert policy.resolve(ROLE_DEEP_REASONING) == deep
    assert policy.resolve(ROLE_LITE) == lite
    assert policy.resolve(ROLE_IMAGE) == deep
    assert ProviderRoutingPolicy.from_dict(policy.to_dict()) == policy


def test_routing_policy_requires_deep_reasoning_role():
    policy = ProviderRoutingPolicy()

    with pytest.raises(ProviderRoleNotConfiguredError):
        policy.resolve(ROLE_DEEP_REASONING)


def test_from_legacy_settings_uses_existing_defaults():
    policy = ProviderRoutingPolicy.from_legacy_settings()

    assert policy.deep_reasoning is not None
    assert policy.lite is not None
    assert policy.deep_reasoning.provider
    assert policy.lite.provider


def test_trading_graph_uses_explicit_routing_policy(monkeypatch):
    created = []
    policy = ProviderRoutingPolicy(
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

    def fake_create_llm(self, provider, model, temperature):
        created.append((provider, model, temperature))
        return {"provider": provider, "model": model, "temperature": temperature}

    monkeypatch.setattr("quantagent.trading_graph.TechnicalTools", lambda: object())
    monkeypatch.setattr("quantagent.trading_graph.SetGraph", DummySetGraph)
    monkeypatch.setattr(TradingGraph, "_create_llm", fake_create_llm)

    graph = TradingGraph(routing_policy=policy)

    assert created == [
        ("openai", "gpt-4o-mini", 0.15),
        ("anthropic", "claude-haiku-4-5-20251001", 0.05),
    ]
    assert graph._routing_policy == policy


def test_strategy_assembler_preserves_routing_policy_from_snapshot(monkeypatch):
    captured = {}
    policy = ProviderRoutingPolicy(
        deep_reasoning=ProviderRoleConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            temperature=0.1,
        ),
        lite=ProviderRoleConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.2,
        ),
    )

    class DummyComponent:
        def __init__(self, *args, **kwargs):
            pass

    class DummyTradingGraph:
        def __init__(self, *, use_checkpointing, routing_policy):
            captured["use_checkpointing"] = use_checkpointing
            captured["routing_policy"] = routing_policy

    monkeypatch.setattr("quantagent.strategy.assembler.PortfolioManager", DummyComponent)
    monkeypatch.setattr("quantagent.strategy.assembler.PositionSizer", DummyComponent)
    monkeypatch.setattr("quantagent.strategy.assembler.RiskManager", DummyComponent)
    monkeypatch.setattr("quantagent.strategy.assembler.PaperBroker", DummyComponent)
    monkeypatch.setattr("quantagent.strategy.assembler.OrderManager", DummyComponent)
    monkeypatch.setattr("quantagent.strategy.assembler.TradingGraph", DummyTradingGraph)

    resolved = StrategyAssembler.from_snapshot(
        {
            "initial_cash": 100000,
            "routing_policy": policy.to_dict(),
            "use_checkpointing": True,
            "universe": ["AAPL"],
        },
        environment=Environment.BACKTEST,
    )

    components = StrategyAssembler.build_components(resolved, db_session=MagicMock())

    assert isinstance(components.graph, DummyTradingGraph)
    assert captured["use_checkpointing"] is True
    assert captured["routing_policy"] == policy
    assert resolved.routing_policy == policy
