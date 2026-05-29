# QuantAgent-kkj.10 — Design: Config Catalog Seed

**Beads issue:** QuantAgent-kkj.10  
**Requirements:** [QuantAgent-kkj.10-RQ-config-catalog-seed.md](../01_requirements/QuantAgent-kkj.10-RQ-config-catalog-seed.md)

---

## Overview

Five deliverables, kept deliberately small: catalog YAML, seed script, model preset persistence, provider constant, and a DB-free smoke test.

```
config/
└── seed/
    └── base_catalog.yaml          ← versioned source of truth

quantagent/
└── provider_registry.py           ← SUPPORTED_PROVIDERS constant (bridge until kkj.11)

scripts/
└── seed_config_catalog.py         ← idempotent upsert loader

apps/streamlit/views/
└── configuration.py               ← modified: DB read/write for model presets

tests/
└── test_smoke_kkj10_config_catalog.py  ← catalog validation without DB
```

---

## 1. `config/seed/base_catalog.yaml`

### Schema

```yaml
catalog_version: "1.0"
generated_at: "<ISO timestamp>"
profiles:
  - name: <str, unique, snake_case>
    kind: "portfolio" | "risk" | "combined" | "model_preset"
    json_config: <dict>
```

`kind` values map directly to `StrategyConfig.kind` (String column — no migration needed).

### Content

**Portfolio profiles**

| name | purpose | universe (representative) |
|---|---|---|
| `paper_us_base` | Paper trading — US large cap + ETF mix | SPY, QQQ, AAPL, MSFT, GOOGL, META, AMZN |
| `backtest_equities_swing` | Backtesting — US equity swing (multi-day holds) | AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, SPY, QQQ |
| `backtest_crypto_intraday` | Backtesting — crypto intraday | BTC-USD, ETH-USD |

Common portfolio fields:

```yaml
universe: [...]
base_position_pct: 0.05
max_position_pct: 0.10
max_daily_loss_pct: 0.05
purpose: "<human-readable description>"
```

**Risk profiles**

| name | purpose |
|---|---|
| `risk_conservative` | Low drawdown tolerance; tight stops |

```yaml
max_drawdown_pct: 0.10
max_position_size_pct: 0.10
stop_loss_pct: 0.02
take_profit_pct: 0.04
```

**Model presets** (`kind: model_preset`)

Seeded only for the 4 providers QuantAgent actually supports at runtime:

| name | provider | model_name |
|---|---|---|
| `openai_default` | openai | gpt-4o-mini |
| `anthropic_default` | anthropic | claude-haiku-4-5-20251001 |
| `qwen_default` | qwen | qwen3-max |
| `azure_default` | azure | `""` (deployment-specific; placeholder) |

```yaml
json_config:
  provider: <str>
  model_name: <str>
  temperature: 0.1
  purpose: "<description>"
```

`azure_default.model_name` is empty string — the seed loads it as placeholder and the UI/config shows it requires the user to set their deployment name. This avoids a ghost preset while being honest about what Azure needs.

---

## 2. `quantagent/provider_registry.py`

Minimal module — single purpose: shared source of truth for supported providers.

```python
SUPPORTED_PROVIDERS: list[str] = ["openai", "anthropic", "qwen", "azure"]
```

**Coordination note with kkj.11:** `QuantAgent-kkj.11` is designing `quantagent/llm/registry.py`
with full capability metadata. When kkj.11 lands, the import in `configuration.py` changes from
`from quantagent.provider_registry import SUPPORTED_PROVIDERS` to
`from quantagent.llm.registry import supported_providers` and this module is deleted. No leaky
abstraction, minimal coupling.

---

## 3. `scripts/seed_config_catalog.py`

### CLI

```
python scripts/seed_config_catalog.py [--reset] [--db-url URL] [--catalog PATH]
```

- `--reset`: TRUNCATE `strategy_configs` before insert (not CASCADE — scoped to config table only)
- `--db-url`: overrides `DATABASE_URL` env var
- `--catalog`: overrides default path (`config/seed/base_catalog.yaml`)

### Core logic

```
load_catalog(path) → dict                    # PyYAML; validate schema_version present
validate_catalog(data) → None               # raise ValueError on missing fields
upsert_profiles(session, profiles) → int    # returns count of rows upserted
```

**Upsert behavior:**
```
for each entry in profiles:
    existing = query by (name)
    if existing:
        existing.kind = entry.kind
        existing.json_config = entry.json_config
        existing.version += 1
        existing.updated_at = now
    else:
        INSERT new row, version=1
```

The key is `name` (unique column in `strategy_configs`). Kind is also updated in case it drifted.

**Failure modes:**

| Condition | Exit behavior |
|---|---|
| `DATABASE_URL` not set | Print error, `sys.exit(1)` |
| DB unreachable | SQLAlchemy `OperationalError` caught; print clear message, `sys.exit(1)` |
| `catalog_version` missing | `ValueError`, `sys.exit(1)` |
| Entry missing `name`/`kind`/`json_config` | `ValueError` with entry index, `sys.exit(1)` |

### Dependencies

- `PyYAML` — already in project? Check requirements. If absent, add to `requirements.txt`.
- `sqlalchemy` — already available
- No external API calls, no yfinance, no network I/O.

---

## 4. `configuration.py` — Model preset DB persistence

### Changes

**Load path (at view initialization):**

```python
def _load_model_presets_from_db(db) -> dict:
    if not db.ok:
        return {}
    with db.SessionLocal() as s:
        rows = s.query(StrategyConfig).filter_by(kind="model_preset").all()
        return {r.name: r.json_config for r in rows}
```

At view entry, merge DB presets into `session_state`:
```python
db_presets = _load_model_presets_from_db(db)
if db_presets:
    st.session_state.model_presets.update(db_presets)
```

This means: DB wins over the hardcoded `default` preset if DB is available.

**Save path (on "Save preset" button):**

After updating `session_state.model_presets`, also persist to DB:

```python
if db.ok:
    with db.SessionLocal() as s:
        existing = s.query(StrategyConfig).filter_by(name=new_name).one_or_none()
        if existing:
            existing.json_config = {...}
            existing.version = (existing.version or 1) + 1
        else:
            s.add(StrategyConfig(name=new_name, kind="model_preset", json_config={...}))
        s.commit()
```

**Provider list:**

```python
from quantagent.provider_registry import SUPPORTED_PROVIDERS
# Replace hardcoded: provider_options = ["openai", "anthropic", "qwen"]
provider_options = SUPPORTED_PROVIDERS
```

### Backward compatibility

- `session_state.model_presets` keeps the hardcoded `"default"` preset as fallback
- If DB is offline, behavior is identical to today
- If DB has a preset with the same name as the hardcoded default, DB wins

---

## 5. `tests/test_smoke_kkj10_config_catalog.py`

**No DB required.** Pure filesystem + import validation.

### Test cases

```python
def test_catalog_loads():
    # YAML parses without error
    # catalog_version key present

def test_catalog_version_field():
    # catalog_version is a non-empty string

def test_portfolio_profiles_count():
    # at least 3 entries with kind="portfolio"

def test_portfolio_universe_symbols_valid():
    # for each portfolio profile, every symbol in json_config["universe"]
    # is in DataProvider.SYMBOL_MAPPING keys

def test_model_preset_providers_supported():
    # for each model_preset entry, json_config["provider"] is in SUPPORTED_PROVIDERS

def test_no_ghost_providers():
    # no model_preset has provider in {"litellm","github-copilot","azure-foundry"}

def test_all_entries_have_required_fields():
    # name, kind, json_config present in every entry
```

---

## Schema compatibility

`StrategyConfig.kind` is `Column(String(20))`. The new `"model_preset"` value is 12 chars — fits. No Alembic migration required.

---

## Dependencies / risks

| Risk | Mitigation |
|---|---|
| PyYAML not in requirements | Add `pyyaml` to `requirements.txt` in implementer phase; check `pip list` first |
| `DataProvider.SYMBOL_MAPPING` symbols change → smoke test breaks | This is desired — the test is a drift detector by design |
| kkj.11 lands before kkj.10 merges | `provider_registry.py` is a thin shim; kkj.11 can absorb it trivially |
| azure preset needs deployment name | Seed with `model_name: ""` and add `purpose` field explaining requirement |

---

## Implementation order (for implementer phase)

1. `config/seed/base_catalog.yaml` — write catalog (no deps)
2. `quantagent/provider_registry.py` — trivial constant (no deps)
3. `scripts/seed_config_catalog.py` — depends on 1
4. `tests/test_smoke_kkj10_config_catalog.py` — depends on 1, 2, DataProvider import
5. `apps/streamlit/views/configuration.py` — depends on 2; modify load/save paths for model presets

Run smoke test after step 4. Full integration test requires DB; smoke test does not.
