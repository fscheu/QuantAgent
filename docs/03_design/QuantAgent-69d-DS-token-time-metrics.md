# QuantAgent-69d: Tracking de tokens y tiempo de ejecución - Design

## Context
- El repo ya tiene persistencia de ejecuciones en `BacktestRun` y un modelo `Log` para eventos.
- Los LLMs se instancian vía LangChain (`ChatOpenAI`, `ChatAnthropic`, `ChatQwen`, `AzureChatOpenAI`).
- La medición de tokens/time debe capturarse **en el borde** de la llamada LLM para evitar inferencias.

## Affected Components
- **LLM invocation layer** (donde se ejecuta `invoke/ainvoke` de LangChain)
- **Backtest** (asociar métricas a `backtest_run_id`)
- **Storage / DB models + migrations**
- **UI/servicios** de consulta de métricas (si existe pantalla de backtest)

## Instrumentation Approach
### 1) LangChain callbacks (preferred)
Implementar un `CallbackHandler` (LangChain) que capture:
- `on_llm_start` → tomar `t0` (perf_counter)
- `on_llm_end` → `duration_ms`, extraer usage/tokens desde el resultado (`response_metadata`, `llm_output`, etc.)
- `on_llm_error` → `duration_ms`, status=error

Notas:
- La extracción de tokens debe ser por provider (OpenAI/Azure suelen exponer usage; Anthropic/Qwen varía según versión).
- `operation` debe mapearse a: nombre de nodo LangGraph / tipo de agente / caller (ej: `indicator_agent`, `pattern_agent`, `trend_agent`, `decision_agent`, `graph_llm`).

### 2) Context propagation
El handler debe recibir (o poder resolver) contexto:
- `environment`
- `symbol`
- `thread_id` / `checkpoint_id`
- `backtest_run_id` (cuando aplica)

Mecanismos posibles (mantener minimalismo en implementación):
- Pasar un objeto `MetricsContext` al construir el graph/backtest.
- O usar `RunnableConfig`/`configurable` de LangChain para transportar metadata.

## Persistence
### Option A (recommended): Dedicated metrics tables
Crear tablas separadas para métricas (más query-friendly que `logs`).

**Table: `llm_call_metrics`** (1 fila por llamada)
- `id`
- `timestamp`
- `provider`, `model`, `operation`
- `input_tokens`, `output_tokens`, `total_tokens`
- `duration_ms`
- `environment`, `symbol`
- `thread_id`, `checkpoint_id`
- `backtest_run_id` (nullable FK a `backtest_runs.id`)
- `extra_data` (JSONB)

**Table: `llm_usage_aggregates`** (agregados por clave)
- `id`
- `scope_type`: `session|backtest`
- `scope_id`: `thread_id` o `backtest_run_id` (string)
- `provider`, `model`, `operation` (nullable para “global”)
- `calls`, `input_tokens_sum`, `output_tokens_sum`, `total_tokens_sum`
- `duration_ms_sum`, `duration_ms_avg`, `duration_ms_p95` (p95 opcional si es simple de mantener)
- `updated_at`

Índices mínimos:
- `llm_call_metrics(backtest_run_id, timestamp)`
- `llm_call_metrics(thread_id, timestamp)`
- `llm_call_metrics(operation)`

### Option B: Use `logs` table
Persistir como eventos `Log(event_type="llm_call")` con `extra_data` conteniendo tokens y duration.

Tradeoff: menos migrations pero queries/aggregations más costosas y menos tipadas.

(Ver decisión en `docs/04_decisions/QuantAgent-69d-DC-metrics-storage.md`)

## Aggregation strategy
- **Write path**: persistir el evento atómico `llm_call_metrics`.
- **Aggregate path**:
  - Batch: al final de backtest, calcular agregados y persistir en `llm_usage_aggregates`.
  - Online (paper/prod): actualizar agregados por `thread_id` periódicamente o al cierre de sesión.

## Exposing metrics
- Servicio/DAO:
  - `get_llm_calls(backtest_run_id=..., thread_id=..., filters...)`
  - `get_llm_aggregates(backtest_run_id=... | thread_id=...)`
- UI (si aplica): en detalle de BacktestRun, sumarizar:
  - total tokens, total duration
  - breakdown por operation + provider/model

## Risks
- Inconsistencias de campos usage entre providers → registrar NULLs y mantener duration.
- Overhead de escritura en DB por llamada → permitir deshabilitar persistencia o agrupar escrituras (si fuera necesario).
- Context propagation incompleta (thread_id/backtest_run_id) → definir claramente la fuente de esos ids.
