# QuantAgent-kkj.11 — Design: Multi-Provider Routing by Role

**Beads issue:** QuantAgent-kkj.11  
**Requirements:** [QuantAgent-kkj.11-RQ-multi-provider-routing.md](../01_requirements/QuantAgent-kkj.11-RQ-multi-provider-routing.md)

---

## Core Design Principle

The issue notes that "catalog of supported providers" and "routing policy by role" must be kept separate. This design treats them as two distinct modules with a clean interface between them.

```
quantagent/llm/
├── __init__.py
├── registry.py     ← Provider catalog (what exists and what it can do)
├── roles.py        ← Role config data model
└── routing.py      ← Routing policy (how roles map to provider configs)
```

---

## Module: `quantagent/llm/registry.py`

### Purpose

Single source of truth for which providers QuantAgent supports at runtime and what capabilities they offer. The UI, settings validation, and routing all import from here.

### Data Model

```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class ProviderCapability:
    provider: str                       # "openai", "anthropic", "qwen", "azure"
    supports_vision: bool               # Can process image inputs
    supports_structured_output: bool    # Supports JSON schema output
    cost_tier: str                      # "cheap" | "mid" | "expensive"
    default_reasoning_model: str        # Recommended for deep_reasoning role
    default_lite_model: str             # Recommended for lite role
    default_vision_model: str           # Recommended for image role
    capability_tags: List[str] = field(default_factory=list)
```

### Registry Contents (initial)

| Provider    | Vision | Structured | Tier      | Reasoning default            | Lite default         | Vision default          |
|-------------|--------|-----------|-----------|------------------------------|----------------------|-------------------------|
| `openai`    | yes    | yes       | mid       | `gpt-4o`                     | `gpt-4o-mini`        | `gpt-4o`                |
| `anthropic` | yes    | yes       | expensive | `claude-haiku-4-5-20251001`  | `claude-haiku-4-5-20251001` | `claude-haiku-4-5-20251001` |
| `qwen`      | yes    | yes       | mid       | `qwen3-max`                  | `qwen3-max`          | `qwen3-vl-plus`         |
| `azure`     | yes    | yes       | mid       | (from deployment)            | (from deployment)    | (from deployment)       |

### Public API

```python
PROVIDER_REGISTRY: Dict[str, ProviderCapability]

def supported_providers() -> List[str]:
    """Return sorted list of supported provider names."""

def get_capability(provider: str) -> ProviderCapability:
    """Raise ValueError if provider not in registry."""
```

---

## Module: `quantagent/llm/roles.py`

### Purpose

Data model for a single role's configuration. Decoupled from the registry — this is the "policy" side.

### Data Model

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ProviderRoleConfig:
    provider: str
    model_name: str
    temperature: float = 0.1
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    capability_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "ProviderRoleConfig": ...

    def validate_against_registry(self) -> None:
        """Raise ValueError if provider not in PROVIDER_REGISTRY."""
```

---

## Module: `quantagent/llm/routing.py`

### Purpose

Aggregates role configs into a policy object that `TradingGraph` and other consumers can use to resolve LLMs by role name.

### Fallback Chain

```
image  →  deep_reasoning  →  lite  →  ProviderRoleNotConfiguredError
```

The resolution order is intentional: vision tasks are more likely to be served by a powerful reasoning model than by a cheap lite model.

### Data Model

```python
from dataclasses import dataclass
from typing import Optional
from quantagent.llm.roles import ProviderRoleConfig

ROLE_DEEP_REASONING = "deep_reasoning"
ROLE_LITE = "lite"
ROLE_IMAGE = "image"

class ProviderRoleNotConfiguredError(ValueError):
    """Raised when a required role cannot be resolved from the routing policy."""

@dataclass
class ProviderRoutingPolicy:
    deep_reasoning: Optional[ProviderRoleConfig] = None
    lite: Optional[ProviderRoleConfig] = None
    image: Optional[ProviderRoleConfig] = None

    def resolve(self, role: str) -> ProviderRoleConfig:
        """
        Resolve role to config. Applies fallback chain if role not set.
        Raises ProviderRoleNotConfiguredError if nothing resolves.
        """

    def resolve_or_none(self, role: str) -> Optional[ProviderRoleConfig]:
        """Same as resolve() but returns None instead of raising."""

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "ProviderRoutingPolicy": ...

    @classmethod
    def from_legacy_settings(cls) -> "ProviderRoutingPolicy":
        """
        Build policy from AGENT_LLM_PROVIDER / GRAPH_LLM_PROVIDER.
        Maps: graph → deep_reasoning, agent → lite, image → None (falls back to deep_reasoning).
        """
```

### Legacy Mapping

```
settings.GRAPH_LLM_PROVIDER / GRAPH_LLM_MODEL / GRAPH_LLM_TEMPERATURE
  → ProviderRoleConfig for deep_reasoning

settings.AGENT_LLM_PROVIDER / AGENT_LLM_MODEL / AGENT_LLM_TEMPERATURE
  → ProviderRoleConfig for lite

image = None  (resolve("image") will fall back to deep_reasoning)
```

---

## Integration: `quantagent/trading_graph.py`

### Changes

`TradingGraph.__init__` gains an optional `routing_policy` parameter:

```python
def __init__(
    self,
    use_checkpointing: bool = False,
    routing_policy: Optional[ProviderRoutingPolicy] = None,
):
    if routing_policy is None:
        routing_policy = ProviderRoutingPolicy.from_legacy_settings()
    self._routing_policy = routing_policy

    self.agent_llm = self._create_llm_from_config(routing_policy.resolve(ROLE_LITE))
    self.graph_llm = self._create_llm_from_config(routing_policy.resolve(ROLE_DEEP_REASONING))
    # image_llm resolved on demand in vision agents
```

A new internal method:
```python
def _create_llm_from_config(self, config: ProviderRoleConfig) -> BaseChatModel:
    """Create LLM from ProviderRoleConfig, replacing the provider/model/temperature tuple."""
```

The existing `_create_llm(provider, model, temperature)` stays but delegates to the new method.

Logging is updated to include `role`:
```python
logger.info("LLM config", extra={
    "event_type": "llm_config",
    "extra_data": {
        "deep_reasoning": {"provider": ..., "model": ..., "role": "deep_reasoning"},
        "lite": {"provider": ..., "model": ..., "role": "lite"},
    }
})
```

### Backward Compatibility

`TradingGraph()` with no arguments continues to work unchanged — `from_legacy_settings()` is called implicitly.

---

## Integration: `quantagent/strategy/assembler.py`

`ResolvedConfig` gets a new field:

```python
@dataclass(frozen=True)
class ResolvedConfig:
    ...existing fields...
    routing_policy: Optional[ProviderRoutingPolicy] = None
```

`StrategyAssembler.assemble()` passes `routing_policy` to `TradingGraph(routing_policy=...)` if set.

---

## Persistence

`StrategyConfig` already supports `kind` discriminators. We add `kind="provider_routing"` to reuse existing infrastructure without a new DB model.

```
StrategyConfig row:
  kind = "provider_routing"
  name = "cost_efficient_default"   (user-defined preset name)
  json_config = { serialized ProviderRoutingPolicy dict }
```

Loading:
```python
raw = session.query(StrategyConfig).filter_by(kind="provider_routing", name=name).one_or_none()
if raw:
    policy = ProviderRoutingPolicy.from_dict(raw.json_config)
```

No new Alembic migration required — uses existing `json_config` column.

---

## Traceability

### BacktestRun metadata extension

After a run completes, the `BacktestRun.extra_data` (or equivalent JSON metadata column) is updated with:

```json
{
  "provider_roles_used": {
    "deep_reasoning": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "role": "deep_reasoning"},
    "lite": {"provider": "openai", "model": "gpt-4o-mini", "role": "lite"},
    "image": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "role": "image", "resolved_from": "deep_reasoning"}
  }
}
```

### Signal metadata

Each `Signal` record's metadata/notes field includes the resolved provider/role used for that analysis cycle.

---

## Configuration UI

`apps/streamlit/views/configuration.py` changes:

1. Provider selector uses `from quantagent.llm.registry import supported_providers` instead of hardcoded list.
2. A new "Provider Routing Preset" section allows loading/saving `ProviderRoutingPolicy` presets by name (stored as `StrategyConfig` with `kind="provider_routing"`).

---

## File Inventory

| File | Action |
|------|--------|
| `quantagent/llm/__init__.py` | New (empty) |
| `quantagent/llm/registry.py` | New |
| `quantagent/llm/roles.py` | New |
| `quantagent/llm/routing.py` | New |
| `quantagent/trading_graph.py` | Modify — add `routing_policy` param, `_create_llm_from_config`, update logging |
| `quantagent/strategy/assembler.py` | Modify — add `routing_policy` field to `ResolvedConfig` |
| `apps/streamlit/views/configuration.py` | Modify — use registry for provider list, add routing preset UI |
| `tests/test_provider_routing.py` | New |
| `tests/test_trading_graph_routing.py` | New (integration) |

---

## Design Decisions

### Why not a new ORM model for routing presets?

Using `StrategyConfig` with `kind="provider_routing"` avoids a new migration and leverages existing CRUD infrastructure. A dedicated model adds no meaningful benefit at this scale.

### Why fallback `image → deep_reasoning → lite`?

Vision tasks require multimodal capability, which is more likely in a larger/more capable model. The lite model may not support vision at all. The `registry.py` `supports_vision` flag is available for stricter validation if needed.

### Why keep `_create_llm(provider, model, temperature)` unchanged?

It's called from `refresh_llms()` and `update_api_key()` code paths that are tested and work today. The new `_create_llm_from_config()` wrapper adapts the new data model to the existing factory — minimizing churn.
