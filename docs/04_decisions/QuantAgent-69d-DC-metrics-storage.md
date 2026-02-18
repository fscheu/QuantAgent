# QuantAgent-69d: Metrics storage strategy (logs vs dedicated tables)

## Context
Necesitamos persistir métricas (tokens + duration) de llamadas LLM y consultarlas/aggregarlas por sesión (`thread_id`) y por backtest (`backtest_run_id`).

El sistema ya tiene:
- `Log` model / `logs` table (eventos genéricos)
- `BacktestRun` (ejecuciones de backtest)

## Options considered
### Option A — Dedicated metrics tables
- `llm_call_metrics` (atómico por llamada)
- `llm_usage_aggregates` (agregados por scope)

Pros:
- Queries simples y performantes
- Menos ambigüedad en schema
- Facilita dashboards/reportes

Cons:
- Requiere migration + modelos nuevos

### Option B — Reusar `logs` table
Persistir todo como `Log(event_type="llm_call")` con `extra_data`.

Pros:
- Cero tablas nuevas
- Reusa infraestructura existente

Cons:
- Queries/aggregations más complejas (JSONB)
- Mayor riesgo de inconsistencias de payload
- Difícil mantener agregados y constraints

## Decision
Elegir **Option A (Dedicated metrics tables)**.

## Consequences
- Se agrega schema específico para métricas.
- `logs` sigue siendo para eventos generales; métricas de costo quedan tipadas.
- El fallback a logs puede evaluarse en el futuro si hay ejecuciones sin DB.

## Status
Accepted
