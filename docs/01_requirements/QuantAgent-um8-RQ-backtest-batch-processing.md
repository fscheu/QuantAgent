# QuantAgent-um8 — Requirements: Batch processing para llamadas de backtesting

## Objetivo
Reducir costo y mejorar throughput de backtests grandes usando APIs de **batch** (asíncronas) de providers LLM (principalmente OpenAI y Anthropic), manteniendo **trazabilidad por request** y tolerando **errores parciales**.

Nivel de detalle: **STANDARD**.

## Contexto
Hoy el backtesting ejecuta análisis LLM de forma síncrona por iteración (candles/activos), lo que:
- escala mal en costo/latencia
- consume rate limits del endpoint síncrono
- dificulta ejecutar backtests masivos (muchas evaluaciones independientes)

Los providers ofrecen endpoints de batch con:
- ~50% descuento (según provider)
- límites de rate dedicados
- turnaround “hasta 24h” (típicamente minutos)

## Alcance
### En alcance
1. **Modo batch** opcional para backtesting (no afecta trading en vivo).
2. Soporte inicial para providers:
   - **OpenAI Batch API** (requests a `/v1/responses` o `/v1/chat/completions` según lo que use el repo).
   - **Anthropic Message Batches** (si se usa Anthropic como provider en backtests).
3. Identificar y batch-ear **unidades de trabajo independientes** en backtesting:
   - múltiples *evaluaciones de estrategia* independientes (p.ej. distintas ventanas/activos/param sets)
   - y/o múltiples invocaciones de decisión por candle **cuando no haya dependencia entre requests**.
4. Parámetros configurables de batching (con defaults):
   - `batch_enabled` (bool)
   - `batch_size` (int)
   - `batch_flush_timeout_sec` (int)
   - `batch_max_in_flight` (int)
   - `batch_poll_interval_sec` (int)
   - `batch_completion_window` (string si aplica; en OpenAI es `24h`)
5. **Trazabilidad**: cada request del batch debe tener un `custom_id` estable que permita mapear resultado ⇄ (backtest_run_id, symbol, timeframe, candle_index, agent_kind/step).
6. Manejo de **errores parciales**:
   - si algunas requests fallan, el backtest debe:
     - registrar error por request
     - continuar con el resto
     - definir política: `fail_fast=false` por default (backtest no aborta todo el batch)

### Fuera de alcance
- Cambiar prompts/estrategia de trading o resultados “business”.
- Batching para ejecución en vivo.
- Soporte de batch para Qwen u otros providers (solo documentar si no hay soporte).
- Optimización de DB o de DataProvider (no relacionado a LLM batch).

## Requisitos funcionales
### RQ-1 — Selector de modo de ejecución
El backtesting debe poder ejecutar LLM en modo:
- `sync` (comportamiento actual)
- `batch` (nuevo)

### RQ-2 — Agrupación y despacho
Dado un stream de “requests LLM” generados por el backtest,
Cuando `batch_enabled=true`,
Entonces el sistema debe agrupar requests en batches según `batch_size` o `batch_flush_timeout_sec` (lo que ocurra primero), y despachar a provider.

### RQ-3 — Paralelismo controlado
Dado que un backtest puede generar múltiples batches,
Cuando se esté en modo batch,
Entonces el sistema debe permitir hasta `batch_max_in_flight` batches en progreso en paralelo (configurable).

### RQ-4 — Correlación request/resultado
Cada request en el batch debe incluir un identificador (`custom_id` o equivalente) que permita reconstruir:
- inputs relevantes (sin necesidad de re-hidratar todo el prompt)
- output (respuesta) y status
- metadatos de provider (modelo, tokens si disponible)

### RQ-5 — Persistencia mínima de resultados
Los resultados de batch deben persistirse en el mismo modelo de trazabilidad que el flujo actual usa para backtest (o extenderlo mínimamente), incluyendo:
- status por request: `completed | failed | cancelled | expired`
- error normalizado (code/message/provider)
- timestamps (submitted/started/completed)

### RQ-6 — Compatibilidad y fallback
- Si `batch_enabled=false` → comportamiento actual.
- Si el provider no soporta batch → el backtest debe fallar con error claro (no silent fallback) salvo que se configure explícitamente `batch_allow_fallback_to_sync=true`.

## Restricciones
- El modo batch es **asíncrono**: se admite esperar/pollear hasta completar.
- Tamaño de input file / requests por batch deben respetar límites del provider.
- No mezclar modelos distintos en el mismo batch cuando el provider lo requiera.

## Definición de Done
- Documentación (`DS/PL/AC`) creada y enlazada.
- Backtesting puede ejecutar un set de evaluaciones en modo `batch` (OpenAI) con resultados trazables.
- Errores parciales no rompen la corrida completa (por default) y quedan auditables.
