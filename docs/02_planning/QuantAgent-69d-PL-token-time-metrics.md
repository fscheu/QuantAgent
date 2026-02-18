# QuantAgent-69d: Tracking de tokens y tiempo de ejecución - Plan

## Level of detail
STANDARD

## Dependencies / References
- RQ: `docs/01_requirements/QuantAgent-69d-RQ-token-time-metrics.md`
- DS: `docs/03_design/QuantAgent-69d-DS-token-time-metrics.md`
- DC: `docs/04_decisions/QuantAgent-69d-DC-metrics-storage.md`
- AC: `docs/05_acceptance_tests/QuantAgent-69d-AC-token-time-metrics.md`

## Implementation tasks (0.5–2h each)
1. **Repo reconnaissance**: identificar el punto único de invocación LLM (LangGraph/agents) y cómo obtener `thread_id`/`checkpoint_id` en runtime.
2. **DB schema + ORM**:
   - Agregar modelos SQLAlchemy: `LLMCallMetric`, `LLMUsageAggregate`.
   - Crear migration Alembic para tablas + índices + FK opcional a `backtest_runs`.
3. **Metrics callback handler**:
   - Implementar `LangChain CallbackHandler` para on_llm_start/end/error.
   - Normalizar extracción de usage tokens (OpenAI/Azure/Anthropic/Qwen) → campos opcionales.
4. **Context propagation**:
   - Definir estructura `MetricsContext` y plomería desde Backtest/TradingGraph.
   - Garantizar `backtest_run_id` disponible durante backtest.
5. **Aggregation**:
   - Implementar función de agregación por `backtest_run_id` al finalizar el backtest.
   - Implementar agregación por `thread_id` (simple: on-demand query + group-by, o materializado si es barato).
6. **Expose metrics**:
   - Servicio/DAO para queries (calls + aggregates).
   - UI (si hay vista de backtest): sumarizar tokens/duration y breakdown por operation.
7. **Manual validation** (sin tests automáticos en esta tarea):
   - Ejecutar un backtest corto y verificar que:
     - se escriben filas en `llm_call_metrics`
     - agregados por backtest coinciden con sumatoria

## Rollout
- Feature flag via env var (por ejemplo `METRICS_ENABLED=true`) para poder activar/desactivar sin cambios de código.
- Activar por defecto sólo en backtest (menor riesgo de performance en prod).

## Risks / Mitigations
- **Escrituras DB por llamada** → permitir batch insert o deshabilitar si impacta performance.
- **Tokens missing por provider** → aceptar NULLs (AC cubre el caso).
- **Context IDs** (thread_id) no disponible en algunos flujos → definir fallback y documentarlo.

## Open Questions
1. ¿La “sesión” para paper/prod debe ser `thread_id` (LangGraph) o una entidad propia (ej. `trading_session_id`)?
2. ¿Querés sólo tokens+tiempo, o también cálculo de costo en USD por modelo (requiere tabla de pricing/versionado)?
3. ¿La UI objetivo es Streamlit (apps/streamlit) o sólo exposición via queries/CSV export?
