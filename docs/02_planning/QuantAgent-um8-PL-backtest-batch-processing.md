# QuantAgent-um8 — Plan: Batch processing para llamadas de backtesting

Nivel de detalle: **STANDARD**.

## Objetivo
Entregar modo batch para backtesting con soporte OpenAI (mínimo viable) y diseño extensible a Anthropic, con trazabilidad y manejo de errores parciales.

## Supuestos
- El modo batch se usa solo en entornos offline (backtesting).
- El repo puede incorporar SDKs oficiales (OpenAI/Anthropic) si no existen ya, o usar HTTP directo si es consistente con el repo.

## Plan de trabajo (tareas ~0.5–2h)

### P0 — Spike / discovery (bloqueante)
1. Identificar “unidad de request” a batch-ear en backtesting
   - localizar dónde se construyen prompts/mensajes por step
   - decidir: batch por step (indicator/pattern/trend/decision) vs batch por evaluación completa
2. Confirmar endpoints batch a usar:
   - OpenAI: `/v1/responses` vs `/v1/chat/completions` (alineado a modelos actuales del repo)
   - Anthropic: Message Batches API (si aplica)
3. Validar restricciones relevantes:
   - modelo único por batch (OpenAI)
   - límites de archivo / requests por batch

### P1 — Diseño de API interna (executor)
4. Definir `BacktestLLMRequest` y `BacktestLLMResult` (data structures)
5. Definir interfaz `BacktestLLMExecutor` y seleccionar ubicación en el repo (módulo nuevo o existente)

### P2 — Implementación OpenAI batch (MVP)
6. Implementar builder de input `.jsonl` con `custom_id` estable
7. Implementar flujo:
   - upload file (purpose=batch)
   - create batch (completion_window=24h)
   - poll status
   - download output file
8. Parse de output JSONL a `BacktestLLMResult[]`
9. Persistencia mínima (según RQ-5) + logging estructurado

### P3 — Integración con backtesting
10. Agregar configuración de batching al snapshot/config del backtest
11. Integrar `sync|batch` en el path de backtesting (sin impactar live)
12. Implementar control de `batch_size`, `flush_timeout`, `max_in_flight`, `poll_interval`

### P4 — Errores parciales y UX de corrida
13. Implementar política `fail_fast` y `batch_allow_fallback_to_sync`
14. Reporte final del backtest incluye conteos: invocations_total, batched_total, failed_total

### P5 — Anthropic (opcional, si el repo realmente lo usa en backtest)
15. Implementar `AnthropicBatchExecutor` siguiendo el mismo contrato
16. Alinear mapeo de errores y trazabilidad

### P6 — Validación
17. Ejecutar un backtest representativo en modo sync y batch y comparar:
   - que complete
   - que deje trazabilidad por request
   - que maneje fallas parciales sin abortar (si se induce una falla)

## Checkpoints / entregables
- Checkpoint A: spike concluido + decisión de unidad de batching
- Checkpoint B: OpenAI batch MVP integrable en backtest
- Checkpoint C: ACs verificables manualmente

## Riesgos y mitigaciones
- Si no se puede reconstruir request por step sin re-escribir prompts:
  - optar por batching a nivel evaluación completa (1 request = 1 evaluación) o limitar el modo batch a un subset.
- Si el tiempo de espera es alto:
  - reducir batch_size y aumentar max_in_flight; mostrar progreso (conteos).

## Cómo validar (manual)
- Ejecutar backtest con `batch_enabled=false` y luego `true` con un conjunto pequeño.
- Verificar que existan artifacts/logs/persistencia para correlacionar `custom_id` → resultado.
