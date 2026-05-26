# QuantAgent-kkj.11 — Plan: Multi-Provider Routing by Role

**Beads issue:** QuantAgent-kkj.11  
**Estimate:** 960 min (~16 h)  
**Requirements:** [QuantAgent-kkj.11-RQ-multi-provider-routing.md](../01_requirements/QuantAgent-kkj.11-RQ-multi-provider-routing.md)  
**Design:** [QuantAgent-kkj.11-DS-multi-provider-routing.md](../03_design/QuantAgent-kkj.11-DS-multi-provider-routing.md)  
**Acceptance:** [QuantAgent-kkj.11-AC-multi-provider-routing.md](../05_acceptance_tests/QuantAgent-kkj.11-AC-multi-provider-routing.md)

---

## Overview

This plan implements multi-provider routing by role in five sequential phases. Each phase is independently committable and testable. Phases 1–3 form the core invariant; Phases 4–5 add persistence and traceability.

**Prerequisite:** No hard dependency on QuantAgent-kkj.10 for phases 1–3. Phase 4 (persistence) coordinates with kkj.10 because both extend `StrategyConfig` usage. If kkj.10 lands first, the `kind="provider_routing"` slot is immediately available.

---

## Phase 1 — Core Abstractions (~3 h)

**Goal:** New `quantagent/llm/` package with registry, role config, and routing policy. No changes to existing behaviour.

### Files to create

```
quantagent/llm/__init__.py              # empty
quantagent/llm/registry.py             # PROVIDER_REGISTRY, supported_providers(), get_capability()
quantagent/llm/roles.py                # ProviderRoleConfig
quantagent/llm/routing.py              # ProviderRoutingPolicy, ProviderRoleNotConfiguredError
```

### Key invariants

- `registry.py` must not import from `trading_graph.py` (no circular imports).
- `routing.py` imports from `roles.py` and `registry.py` only.
- `ProviderRoutingPolicy.from_legacy_settings()` imports `quantagent.settings` at call time (lazy).

### Tests to write

`tests/test_provider_routing.py` covering AC-01 through AC-07 and AC-10.  
All tests must pass without real API keys (no LLM calls).

### Validation command

```bash
pytest tests/test_provider_routing.py -v
python -m compileall -q quantagent/llm/
```

---

## Phase 2 — TradingGraph Wiring (~2.5 h)

**Goal:** `TradingGraph` accepts an optional `routing_policy` and logs resolved roles. Backward compat fully preserved.

### Files to modify

```
quantagent/trading_graph.py
```

### Changes

1. Add `routing_policy: Optional[ProviderRoutingPolicy] = None` parameter to `__init__`.
2. Add `_create_llm_from_config(config: ProviderRoleConfig) -> BaseChatModel` internal method.
3. Call `from_legacy_settings()` when `routing_policy is None`.
4. Update `refresh_llms()` to respect `self._routing_policy`.
5. Update `llm_config` log event to include `role` key for each LLM.

### Backward compat test

```bash
# Must still work without any routing_policy argument:
python -c "from quantagent.trading_graph import TradingGraph; print('OK')"
```

### Tests to write

`tests/test_trading_graph_routing.py` covering AC-08 and AC-09 (with mocked `_create_llm`).

### Validation command

```bash
pytest tests/test_trading_graph_routing.py -v
pytest tests/ -k "trading_graph" --tb=short
```

---

## Phase 3 — Strategy-Level Wiring (~1.5 h)

**Goal:** `ResolvedConfig` and `StrategyAssembler` can carry a routing policy through to `TradingGraph`.

### Files to modify

```
quantagent/strategy/assembler.py
```

### Changes

1. Add `routing_policy: Optional[ProviderRoutingPolicy] = None` field to `ResolvedConfig`.
2. Update `StrategyAssembler.assemble()` to pass `routing_policy` when creating `TradingGraph`.
3. Existing callers that do not pass `routing_policy` continue to work (field defaults to None → legacy path).

### Validation command

```bash
pytest tests/ -x --tb=short
```

---

## Phase 4 — Persistence (~2.5 h)

**Goal:** Named routing presets are persistible in DB and editable in the Streamlit Configuration UI.

### Coordinates with QuantAgent-kkj.10

If kkj.10 has landed, `StrategyConfig` with new `kind` values is already in use. No additional migration needed — just add `kind="provider_routing"` usage.

### Files to modify

```
apps/streamlit/views/configuration.py
```

### Changes

1. Replace hardcoded provider list (`["openai", "anthropic", "qwen"]`) with `supported_providers()` from registry.
2. Add "Provider Routing Preset" section:
   - Preset name input
   - Role selector (deep_reasoning / lite / image) with provider + model dropdowns per role
   - Save / Load / Delete buttons wired to DB via `StrategyConfig(kind="provider_routing")`
3. Loaded preset populates the strategy assembler's `routing_policy` for the current session.

### No new DB migration required

`StrategyConfig` `json_config` column already holds arbitrary JSON. `kind="provider_routing"` is a new string value — no schema change.

### Seed entry (coordinate with kkj.10)

Add a default routing preset to `config/seed/base_catalog.yaml` (or the equivalent path defined by kkj.10):

```yaml
provider_routing_presets:
  - name: "cost_efficient_default"
    deep_reasoning:
      provider: "anthropic"
      model_name: "claude-haiku-4-5-20251001"
      temperature: 0.1
      capability_tags: ["reasoning"]
    lite:
      provider: "openai"
      model_name: "gpt-4o-mini"
      temperature: 0.1
      capability_tags: ["cheap"]
    image:
      provider: "anthropic"
      model_name: "claude-haiku-4-5-20251001"
      temperature: 0.1
      capability_tags: ["vision"]
```

### Validation command

```bash
pytest tests/test_trading_graph_routing.py::test_routing_policy_db_persistence -v
# Manual: open Streamlit → Configuration → check provider selector uses registry
```

---

## Phase 5 — Traceability (~2 h)

**Goal:** Every backtest run and paper trading signal records which role/provider/model was used.

### Files to modify

```
quantagent/backtesting/backtest.py
quantagent/trading/scheduler.py
```

### Changes

1. After `BacktestRun` is created, add `provider_roles_used` to its metadata/`extra_data`:

```python
routing_policy = self.assembler.resolved_config.routing_policy or ProviderRoutingPolicy.from_legacy_settings()
run_metadata["provider_roles_used"] = {
    role: config.to_dict() | {"role": role}
    for role, config in routing_policy.to_dict().items()
    if config is not None
}
```

2. For `Signal` records generated by the scheduler, add `provider_role` to the signal's `notes` or `metadata` field.

### Validation command

```bash
pytest tests/test_trading_graph_routing.py::test_backtest_metadata_includes_roles -v
```

---

## Operational Guide: Cost-Efficient Routing

To configure a cost-efficient strategy that uses deep reasoning only where needed:

**Option A — via env vars (backward compat, no DB required)**

```env
GRAPH_LLM_PROVIDER=anthropic
GRAPH_LLM_MODEL=claude-haiku-4-5-20251001
AGENT_LLM_PROVIDER=openai
AGENT_LLM_MODEL=gpt-4o-mini
```

This maps automatically to: `graph → deep_reasoning`, `agent → lite`, `image → fallback to deep_reasoning`.

**Option B — via routing policy (explicit, persistible)**

```python
from quantagent.llm.roles import ProviderRoleConfig
from quantagent.llm.routing import ProviderRoutingPolicy

policy = ProviderRoutingPolicy(
    deep_reasoning=ProviderRoleConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.1,
        capability_tags=["reasoning"],
    ),
    lite=ProviderRoleConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.1,
        capability_tags=["cheap"],
    ),
    image=None,  # falls back to deep_reasoning
)

from quantagent.trading_graph import TradingGraph
graph = TradingGraph(routing_policy=policy)
```

**Option C — via UI preset**

1. Open Streamlit → Configuration → "Provider Routing Preset".
2. Configure roles, name the preset (e.g., `cost_efficient_default`), click Save.
3. Select the preset in the strategy assembler or scheduler config.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Circular import between `llm/routing.py` and `trading_graph.py` | Medium | `from_legacy_settings()` imports `settings` lazily at call time; registry imported at module level from isolated package |
| `azure` provider requires extra env vars not always set | High | `from_legacy_settings()` only uses azure if `AGENT_LLM_PROVIDER=azure` is set; registry notes azure needs additional config |
| StrategyConfig `kind="provider_routing"` clashes with kkj.10 changes | Low | Coordinate before Phase 4 merge; `kind` is a string field with no uniqueness constraint |
| Existing tests break due to TradingGraph signature change | Low | `routing_policy` has a default of `None` — all existing call sites continue to work |

---

## Commit Plan

| Phase | Branch      | Commit message                                                      |
|-------|-------------|----------------------------------------------------------------------|
| 1     | feature/kkj.11-routing | `feat(kkj.11): add provider registry, role config, and routing policy` |
| 2     | feature/kkj.11-routing | `feat(kkj.11): wire routing policy into TradingGraph`               |
| 3     | feature/kkj.11-routing | `feat(kkj.11): propagate routing policy through StrategyAssembler`  |
| 4     | feature/kkj.11-routing | `feat(kkj.11): persist routing presets in DB, use registry in UI`   |
| 5     | feature/kkj.11-routing | `feat(kkj.11): add role/provider traceability to backtest and signals` |

Each phase commit should include the relevant test additions/updates.

Human gate: review + merge to `main` after each phase or as a bundle once all phases pass.
