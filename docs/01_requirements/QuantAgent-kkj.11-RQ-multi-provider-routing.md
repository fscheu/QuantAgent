# QuantAgent-kkj.11 — Requirements: Multi-Provider Routing by Role

**Beads issue:** QuantAgent-kkj.11  
**Labels:** configuration, cost, llm  
**Parent:** QuantAgent-kkj (M2 milestone)  
**Related:** QuantAgent-kkj.10 (config catalog & persistence — must land before or alongside)

---

## Context

QuantAgent currently operates with two global LLM slots (`AGENT_LLM_PROVIDER`, `GRAPH_LLM_PROVIDER`). This makes it impossible to route cheap tasks to a lite model while reserving an expensive reasoning model for complex analysis, or to route vision tasks to a multimodal model. There is also a divergence between the list of providers shown in the UI (`openai`, `anthropic`, `qwen`) and those supported by the runtime (`openai`, `anthropic`, `qwen`, `azure`).

---

## Functional Requirements

### FR-01 — Provider Role Abstraction

The system must support at least three named provider roles:

| Role identifier        | Intended use                                               |
|------------------------|------------------------------------------------------------|
| `deep_reasoning`       | Complex analysis, final trading decision, heavy reasoning  |
| `lite`                 | Lightweight tasks, signal pre-processing, cheap calls      |
| `image`                | Multimodal / vision tasks (chart pattern analysis, trend)  |

Each role is **independent**; multiple roles may resolve to the same provider/model configuration.

### FR-02 — Role Configuration Schema

Each configured role must carry at minimum:

| Field              | Type             | Required | Notes                                      |
|--------------------|------------------|----------|--------------------------------------------|
| `provider`         | `str`            | yes      | Must be a key in the Provider Registry     |
| `model_name`       | `str`            | yes      | Provider-specific model identifier         |
| `temperature`      | `float`          | yes      | Default 0.1                                |
| `timeout_seconds`  | `int`            | no       | Request timeout override                   |
| `max_retries`      | `int`            | no       | Retry count override                       |
| `capability_tags`  | `List[str]`      | no       | e.g. `["reasoning", "cheap", "vision"]`    |

### FR-03 — Central Provider Registry

A single `PROVIDER_REGISTRY` must enumerate all providers QuantAgent supports at runtime, including their capabilities. The UI must derive its provider selector from this registry — no separate hardcoded list.

Providers that must appear in the registry at minimum: `openai`, `anthropic`, `qwen`, `azure`.

Each registry entry must declare at minimum:
- Whether the provider supports vision/multimodal input
- Default model recommendations per role type (reasoning, lite, vision)
- Cost tier hint (`cheap`, `mid`, `expensive`)

### FR-04 — Routing Policy

A `ProviderRoutingPolicy` must aggregate the role configurations into a single persistible unit. It must support:

- **Resolution**: given a role name, return the configured `ProviderRoleConfig`
- **Fallback chain**: if the requested role is not configured, fall back through `image → deep_reasoning → lite`; if none are configured, raise a clear error
- **Serialisation**: `to_dict()` / `from_dict()` for DB persistence

### FR-05 — Backward Compatibility

If only the legacy env vars `AGENT_LLM_PROVIDER` / `GRAPH_LLM_PROVIDER` are set (no explicit role policy), the system must still start. `ProviderRoutingPolicy.from_legacy_settings()` must map them to sensible role defaults:

- `GRAPH_LLM_PROVIDER` / `GRAPH_LLM_MODEL` → `deep_reasoning` role
- `AGENT_LLM_PROVIDER` / `AGENT_LLM_MODEL` → `lite` role
- `image` role falls back to `deep_reasoning` in the legacy case

### FR-06 — Strategy-Level Role Consumption

At least one code path in the system must be able to resolve different roles for different stages, without hardcoding the same provider everywhere. Minimum acceptable scope:

- `TradingGraph` uses `deep_reasoning` for the graph/decision LLM and `lite` for the agent LLM
- Vision agents (pattern, trend) use `image` when configured; fall back to `deep_reasoning` otherwise

### FR-07 — Persistible Configuration

The `ProviderRoutingPolicy` must be storable in the database as a named preset — not only in env vars or `st.session_state`. The Configuration UI must expose read/edit of persisted routing presets.

### FR-08 — Traceability

Every backtest run and every paper trading signal must record in its metadata which role/provider/model was actually used at runtime. This metadata must survive process restarts (i.e., be persisted, not just logged).

### FR-09 — Explicit Fallback Error

When a role cannot be resolved (no configuration, no fallback), the system must raise a named exception (`ProviderRoleNotConfiguredError`) with a message identifying the missing role and suggesting a fix.

---

## Non-Functional Requirements

- **No breaking changes** to the existing `TradingGraph()` constructor call without arguments.
- **No new provider SDKs** are introduced in this ticket; only providers with existing runtime support are registered.
- The `PROVIDER_REGISTRY` is the canonical list; adding a new provider in the future must only require updating the registry, not scattered `if provider == ...` checks across the codebase.

---

## Out of Scope

- Integrating providers that require new SDKs (LiteLLM, GitHub Copilot, Azure Foundry)
- Migrating credentials to a secrets manager
- Automatic cost optimisation based on benchmarked latency/cost
- Redesigning all existing strategies to use multi-role routing

---

## Dependencies / Related Issues

- **QuantAgent-kkj.10**: Provides DB persistence for model presets. The `ProviderRoutingPolicy` persistence layer coordinates with the `kind="provider_routing"` extension of `StrategyConfig`.
- **QuantAgent-kkj** (parent): M2 milestone tracking.
