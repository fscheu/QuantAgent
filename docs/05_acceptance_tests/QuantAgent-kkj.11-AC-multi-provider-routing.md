# QuantAgent-kkj.11 — Acceptance Criteria: Multi-Provider Routing by Role

**Beads issue:** QuantAgent-kkj.11  
**Requirements:** [QuantAgent-kkj.11-RQ-multi-provider-routing.md](../01_requirements/QuantAgent-kkj.11-RQ-multi-provider-routing.md)  
**Design:** [QuantAgent-kkj.11-DS-multi-provider-routing.md](../03_design/QuantAgent-kkj.11-DS-multi-provider-routing.md)

---

## AC-01 — Role abstraction module exists and is importable

**Validates:** FR-01, FR-02

```
Given a standard QuantAgent installation with dependencies installed
When the following import is executed:
    from quantagent.llm.routing import ProviderRoutingPolicy, ROLE_DEEP_REASONING, ROLE_LITE, ROLE_IMAGE
    from quantagent.llm.roles import ProviderRoleConfig
    from quantagent.llm.registry import PROVIDER_REGISTRY, supported_providers
Then no ImportError is raised and all names resolve
```

**Automated test:** `tests/test_provider_routing.py::test_imports`

---

## AC-02 — Provider registry is the single source of truth

**Validates:** FR-03

```
Given the PROVIDER_REGISTRY in quantagent.llm.registry
When supported_providers() is called
Then it returns a list containing at least ["anthropic", "azure", "openai", "qwen"] (sorted)

Given the Streamlit configuration UI
When the provider selector renders
Then the options are derived from supported_providers() and not from a hardcoded list
```

**Automated test:** `tests/test_provider_routing.py::test_registry_contains_required_providers`

---

## AC-03 — ProviderRoleConfig round-trips through dict serialisation

**Validates:** FR-02, FR-07

```
Given a ProviderRoleConfig instance with provider="anthropic", model_name="claude-haiku-4-5-20251001",
      temperature=0.1, capability_tags=["reasoning"]
When config.to_dict() is called and then ProviderRoleConfig.from_dict() is applied to the result
Then the resulting object equals the original
```

**Automated test:** `tests/test_provider_routing.py::test_role_config_roundtrip`

---

## AC-04 — ProviderRoutingPolicy resolves configured roles

**Validates:** FR-04

```
Given a ProviderRoutingPolicy with:
    deep_reasoning = ProviderRoleConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001", temperature=0.1)
    lite = ProviderRoleConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.1)
    image = None
When policy.resolve(ROLE_DEEP_REASONING) is called
Then it returns the anthropic ProviderRoleConfig

When policy.resolve(ROLE_LITE) is called
Then it returns the openai ProviderRoleConfig
```

**Automated test:** `tests/test_provider_routing.py::test_resolve_configured_roles`

---

## AC-05 — Fallback chain: image → deep_reasoning

**Validates:** FR-04, FR-06

```
Given a ProviderRoutingPolicy with:
    deep_reasoning = ProviderRoleConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001", temperature=0.1)
    lite = None
    image = None
When policy.resolve(ROLE_IMAGE) is called
Then it returns the deep_reasoning ProviderRoleConfig (fallback applied)
```

**Automated test:** `tests/test_provider_routing.py::test_image_falls_back_to_deep_reasoning`

---

## AC-06 — ProviderRoleNotConfiguredError raised when no fallback available

**Validates:** FR-09

```
Given a ProviderRoutingPolicy with all roles set to None
When policy.resolve(ROLE_DEEP_REASONING) is called
Then ProviderRoleNotConfiguredError is raised
And the exception message identifies the missing role name
```

**Automated test:** `tests/test_provider_routing.py::test_empty_policy_raises_error`

---

## AC-07 — Legacy backward compatibility: system starts without role config

**Validates:** FR-05

```
Given only AGENT_LLM_PROVIDER=openai, AGENT_LLM_MODEL=gpt-4o-mini,
      GRAPH_LLM_PROVIDER=anthropic, GRAPH_LLM_MODEL=claude-haiku-4-5-20251001
      in the environment (no explicit routing policy)
When ProviderRoutingPolicy.from_legacy_settings() is called
Then it returns a ProviderRoutingPolicy where:
    - resolve(ROLE_DEEP_REASONING).provider == "anthropic"
    - resolve(ROLE_LITE).provider == "openai"
    - resolve(ROLE_IMAGE) returns the deep_reasoning config (fallback)
```

**Automated test:** `tests/test_provider_routing.py::test_from_legacy_settings`

---

## AC-08 — TradingGraph initialises without arguments (backward compat)

**Validates:** FR-05

```
Given a valid environment with AGENT_LLM_PROVIDER and GRAPH_LLM_PROVIDER set
When TradingGraph() is instantiated with no arguments
Then no exception is raised
And trading_graph.agent_llm is a BaseChatModel instance
And trading_graph.graph_llm is a BaseChatModel instance
```

**Automated test:** Integration test in `tests/test_trading_graph_routing.py::test_trading_graph_default_init`  
(requires mock LLM or valid API key — use provider patching)

---

## AC-09 — TradingGraph accepts explicit routing policy

**Validates:** FR-06

```
Given a ProviderRoutingPolicy where deep_reasoning=anthropic and lite=openai
When TradingGraph(routing_policy=policy) is instantiated (with mocked LLM factories)
Then graph_llm was created with anthropic config
And agent_llm was created with openai config
```

**Automated test:** `tests/test_trading_graph_routing.py::test_trading_graph_explicit_policy`

---

## AC-10 — ProviderRoutingPolicy round-trips through dict serialisation

**Validates:** FR-07

```
Given a ProviderRoutingPolicy with deep_reasoning and lite both configured
When policy.to_dict() is called and then ProviderRoutingPolicy.from_dict() applied
Then the result equals the original policy
```

**Automated test:** `tests/test_provider_routing.py::test_routing_policy_roundtrip`

---

## AC-11 — Routing policy persistible via StrategyConfig(kind="provider_routing")

**Validates:** FR-07

```
Given a ProviderRoutingPolicy instance and a database session
When the policy is serialised and stored as StrategyConfig(kind="provider_routing", name="test_preset", json_config=policy.to_dict())
And then loaded back via a fresh query and ProviderRoutingPolicy.from_dict()
Then the loaded policy resolves roles identically to the original
```

**Automated test:** `tests/test_trading_graph_routing.py::test_routing_policy_db_persistence`  
(can use SQLite in-memory)

---

## AC-12 — Backtest run metadata includes role/provider/model snapshot

**Validates:** FR-08

```
Given a backtest run executed with an explicit ProviderRoutingPolicy
When the BacktestRun record is read from the database after completion
Then its metadata/extra_data JSON contains a "provider_roles_used" key
And each entry identifies provider, model, and role name
```

**Automated test:** Smoke check in `tests/test_trading_graph_routing.py::test_backtest_metadata_includes_roles`  
(lightweight — does not execute real trades)

---

## AC-13 — Operational documentation exists

**Validates:** FR-01 through FR-08 holistically

```
Given the planning document docs/02_planning/QuantAgent-kkj.11-PL-multi-provider-routing.md
When read by a new engineer
Then it explains:
    - how to configure a cost-efficient routing preset with deep_reasoning + lite + image roles
    - how to persist that preset and seed it for QA/DEV
    - the backward compat guarantee for legacy env-var configs
```

**Manual check** — reviewed by issue owner before close.

---

## Passing Criteria

The issue is ready to merge when:

- [ ] All automated tests pass: `pytest tests/test_provider_routing.py tests/test_trading_graph_routing.py -v`
- [ ] No existing tests regress
- [ ] `TradingGraph()` (no args) continues to work with the same env vars as before
- [ ] The Streamlit Configuration view uses `supported_providers()` from the registry
- [ ] At least one `BacktestRun` record produced by the new code contains `provider_roles_used` metadata
- [ ] AC-13 manual check passes
