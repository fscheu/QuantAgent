# QuantAgent-kkj.10 — RQ: Persistir y seedear catálogo base de configuración para QA/DEV limpio

**Issue:** QuantAgent-kkj.10  
**Type:** Feature / DX / QA  
**Parent:** QuantAgent-kkj (M2 Milestone)  
**Status:** open  
**Labels:** configuration, dx, qa  
**Design:** [QuantAgent-kkj.10-DS-config-catalog-seed.md](../03_design/QuantAgent-kkj.10-DS-config-catalog-seed.md)

---

## Problema

QuantAgent ya dispone de scripts de seeding para datos transaccionales (`scripts/seed_dev.py`,
`scripts/bootstrap_qa_minimal.py`), pero no existe una forma confiable de inicializar un environment
limpio con un catálogo base de configuración utilizable desde la UI.

Gaps concretos verificados en el código actual:

| Gap | Evidencia |
|---|---|
| DB limpia → combos portfolio vacíos en UI | `apps/streamlit/views/configuration.py:282` — `_collect_profiles_from_db` devuelve `[]` cuando no hay filas |
| Model presets viven solo en `session_state` | `apps/streamlit/app.py:65-72`, `views/configuration.py:55-60` |
| UI expone 3 providers; backend soporta 4 | UI: `["openai","anthropic","qwen"]`; `TradingGraph`: también `"azure"` (`trading_graph.py:157-292`) |
| `DEFAULT_SCHEDULER_ASSETS = ["BTC","SPX"]` — universo mínimo | `quantagent/settings.py:110` |

---

## Cambio requerido

Implementar un bootstrap versionado de configuración base. Cinco entregables:

### 1. Catálogo versionado (`config/seed/base_catalog.yaml`)

Fuente de verdad explícita y legible para:
- ≥ 3 portfolio profiles operativos con universo más amplio que el default actual:
  - `paper_us_base` — paper trading US equities
  - `backtest_equities_swing` — backtesting equity swing
  - `backtest_crypto_intraday` — backtesting crypto
- ≥ 1 risk profile base
- Model presets LLM solo para providers realmente soportados: `openai`, `anthropic`, `qwen`, `azure`

### 2. Script de seed idempotente (`scripts/seed_config_catalog.py`)

- Lee `base_catalog.yaml` y hace upsert de todos los items en `strategy_configs`
- Idempotente: si un perfil ya existe con el mismo nombre y kind, actualiza `json_config` y bumps `version`
- Soporta `--reset` para truncar `strategy_configs` antes de insertar
- Falla con mensaje claro si la DB no está accesible o si el schema no es compatible
- No toca tablas de datos transaccionales (orders, trades, signals, etc.)

### 3. Persistencia durable de model presets

Extender el uso de `StrategyConfig` con `kind="model_preset"` para los presets LLM.
No requiere migración de schema (la columna `kind` es `String`, no `Enum`).

La UI (`configuration.py`) debe:
- Leer model presets desde DB si está disponible (en vez de solo `session_state`)
- Escribir presets a DB al guardar (además de `session_state` como fallback)
- Inicializar `session_state` con los presets leídos desde DB al cargar la vista

### 4. Unificación de providers soportados

Crear una constante compartida `SUPPORTED_PROVIDERS` en un módulo lightweight
(`quantagent/provider_registry.py` o inline en `quantagent/settings.py`) con los 4 providers
que el backend ya soporta: `["openai", "anthropic", "qwen", "azure"]`.

La UI debe importar desde esa constante en vez de hardcodear la lista de 3.

**Nota de coordinación con kkj.11:** `QuantAgent-kkj.11` está diseñando un registry completo con
capacidades por provider (`quantagent/llm/registry.py`). La constante introducida aquí debe ser un
puente mínimo que kkj.11 reemplazará. Cuando kkj.11 aterrice, la UI importa de `llm.registry`
en vez de `provider_registry`.

### 5. Smoke test de drift catálogo/schema

Un test (`tests/test_smoke_kkj10_config_catalog.py`) que:
- Lee `base_catalog.yaml` y valida que cada entry tiene los campos mínimos (`name`, `kind`, `json_config`)
- Para `kind="portfolio"`: verifica que `universe` contiene solo símbolos válidos (importados desde `DataProvider.SYMBOL_MAPPING`)
- Para `kind="model_preset"`: verifica que `provider` está en `SUPPORTED_PROVIDERS`
- No requiere DB disponible (valida solo el catálogo YAML)

---

## Acceptance Criteria

- [ ] **AC1:** Existe `config/seed/base_catalog.yaml` con versión, al menos 3 portfolio profiles y al menos 1 model preset por provider soportado.
- [ ] **AC2:** `scripts/seed_config_catalog.py --reset` en una DB limpia se ejecuta sin errores y produce ≥ 5 rows en `strategy_configs`.
- [ ] **AC3:** Tras correr el seed, la UI Configuration muestra portfolios base en los selectores "Paper default portfolio" y "Backtest default portfolio".
- [ ] **AC4:** Los 3 portfolio profiles tienen universo más amplio que `["BTC","SPX"]` y nombre de negocio explícito (`paper_us_base`, `backtest_equities_swing`, `backtest_crypto_intraday`).
- [ ] **AC5:** Model presets se leen desde DB si está disponible; `session_state` es fallback para DB offline.
- [ ] **AC6:** La UI y el backend comparten la misma lista de providers soportados (`openai`, `anthropic`, `qwen`, `azure`); no divergen.
- [ ] **AC7:** Los presets seeded no incluyen providers sin soporte runtime real (`litellm`, `github-copilot`, `azure-foundry`).
- [ ] **AC8:** `pytest tests/test_smoke_kkj10_config_catalog.py` pasa sin DB y falla si el catálogo queda incompatible con el schema o con `SUPPORTED_PROVIDERS`.
- [ ] **AC9:** README o user manual documenta cómo reinicializar QA/DEV con este bootstrap.

---

## Archivos afectados

| Archivo | Cambio |
|---|---|
| `config/seed/base_catalog.yaml` | Nuevo — catálogo versionado |
| `scripts/seed_config_catalog.py` | Nuevo — script de seed idempotente |
| `quantagent/provider_registry.py` | Nuevo — constante `SUPPORTED_PROVIDERS` |
| `apps/streamlit/views/configuration.py` | Modificado — leer/escribir presets desde DB; importar `SUPPORTED_PROVIDERS` |
| `tests/test_smoke_kkj10_config_catalog.py` | Nuevo — smoke test sin DB |

---

## Fuera de scope

- No poblar trades, orders, backtest runs ni heartbeats sintéticos.
- No automatizar la ejecución del seed en deploy/CI por defecto.
- No agregar soporte runtime nuevo para `litellm`, `github-copilot` o `azure-foundry`.
- No rediseñar la UX completa de Configuration más allá de la persistencia de presets.
- No implementar el registry completo de capacidades por provider (eso es kkj.11).
