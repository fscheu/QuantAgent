# QuantAgent-um8 — Acceptance Criteria: Batch processing para backtesting

Nivel de detalle: **STANDARD**.

> Referencias: ver RQ en `docs/01_requirements/QuantAgent-um8-RQ-backtest-batch-processing.md`.

## AC-1 — Modo sync sigue funcionando
**Given** un backtest configurado con `batch_enabled=false`
**When** se ejecuta el backtest
**Then** el flujo de invocaciones LLM se comporta como antes (sin crear jobs batch)

## AC-2 — Se crean batches por tamaño
**Given** un backtest con `batch_enabled=true` y `batch_size = N`
**And** se generan más de N requests LLM independientes
**When** corre el backtest
**Then** los requests se agrupan en batches de tamaño N (salvo el último)

## AC-3 — Flush por timeout
**Given** `batch_enabled=true`, `batch_size` grande, y `batch_flush_timeout_sec = T`
**When** pasan T segundos sin completar un batch por tamaño
**Then** el sistema despacha el batch parcial acumulado

## AC-4 — Límite de batches en vuelo
**Given** `batch_max_in_flight = K`
**When** el backtest genera batches más rápido de lo que el provider los completa
**Then** el sistema no supera K batches en estado “in-progress” simultáneo

## AC-5 — Trazabilidad por request
**Given** un batch con múltiples requests
**When** el provider devuelve resultados
**Then** cada resultado se mapea inequívocamente a su `custom_id`
**And** el `custom_id` permite reconstruir al menos: (backtest_run_id, symbol, timeframe, candle_index, step/agent)

## AC-6 — Errores parciales no abortan por default
**Given** un batch donde algunas requests fallan (por ejemplo, error de validación o rate limit batch)
**And** `fail_fast=false`
**When** se procesan resultados
**Then** el backtest continúa procesando las requests exitosas
**And** los errores quedan registrados y consultables por request

## AC-7 — Provider sin soporte batch falla claramente
**Given** `batch_enabled=true`
**And** un provider configurado que no soporta batch en este repo
**When** se intenta ejecutar backtest
**Then** se devuelve un error explícito indicando “batch no soportado para provider=<x>”

## AC-8 — Parámetros de batching quedan auditables
**Given** un backtest ejecutado con `batch_enabled=true`
**When** se inspecciona el registro del backtest run (o metadata persistida)
**Then** se observan los parámetros efectivos de batching (size/timeout/max_in_flight/poll_interval/completion_window)
