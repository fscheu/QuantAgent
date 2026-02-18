# QuantAgent-um8 — Design: Batch processing para llamadas de backtesting

## Objetivo
Diseñar una integración **mínima** para ejecutar invocaciones LLM de backtesting vía APIs batch de provider, sin tocar el flujo de trading en vivo.

## Drivers
- Costo: aprovechar descuento batch (OpenAI: “50% lower costs” según docs).
- Throughput: pool de rate limits independiente.
- Trazabilidad: correlación 1:1 request→resultado.
- Robustez: errores parciales.

## No-goals
- No re-diseñar prompts/agentes ni la lógica de decisión.
- No “forzar” batch dentro de LangGraph/LangChain si no está soportado nativamente.

## Arquitectura propuesta (alto nivel)
Introducir una capa explícita de ejecución para backtesting:

- **BacktestLLMExecutionMode**: `sync | batch`
- **BacktestLLMExecutor (interface)**:
  - `submit(requests[]) -> job_handle`
  - `poll(job_handle) -> results[]`
  - `materialize(results[]) -> domain objects` (mapeo a lo que el backtest necesita)

Implementaciones:
- `SyncExecutor`: wrap del comportamiento actual (síncrono, por iteración)
- `OpenAIBatchExecutor`: usa OpenAI Batch API
- `AnthropicBatchExecutor`: usa Message Batches API

La salida del executor debe ser lo suficientemente “plana” como para integrarse con el backtest sin re-escribir todo el engine.

## Identificación de unidad de batching
Se define `BacktestLLMRequest` como la **unidad atómica**.
Campos mínimos:
- `custom_id`
- `provider`
- `model`
- `payload` (mensajes/params del endpoint)
- `trace`:
  - `backtest_run_id`
  - `symbol`
  - `timeframe`
  - `candle_index` (o timestamp)
  - `step` (p.ej. indicator|pattern|trend|decision o “strategy_eval”)

**Regla**: solo se batch-ean requests **independientes** (sin dependencia de estado entre sí).

## OpenAI Batch API (referencia)
Según docs oficiales, OpenAI Batch:
- es asíncrono, con ventana `24h`
- requiere input `.jsonl` con líneas `custom_id/method/url/body`
- soporta endpoints como `/v1/responses` y `/v1/chat/completions`

Fuente: https://developers.openai.com/api/docs/guides/batch

### Example (minimal)
_(Ilustrativo; no es implementación completa)_

```jsonl
{"custom_id":"bt:123:SPX:4h:42:decision","method":"POST","url":"/v1/responses","body":{"model":"gpt-4.1-mini","input":"..."}}
```

## Anthropic Message Batches (referencia)
Anthropic provee Message Batches con:
- procesamiento asíncrono
- hasta 10k requests por batch
- costo ~50% menor (según anuncio oficial)

Fuente (anuncio): https://claude.com/blog/message-batches-api

## Manejo de errores parciales
Normalizar resultado por request:
- `status`: completed|failed
- `output`: texto/JSON (según formato)
- `error`: {provider_code, message, retryable?}

Persistencia:
- guardar resultado por `custom_id`
- permitir reintento selectivo de requests fallidas (fuera de alcance implementarlo completo; sí diseñar hooks)

## Trazabilidad / observabilidad
Requisitos mínimos:
- log structured por batch: batch_id, counts (total/completed/failed)
- log por request fallida: custom_id + error normalizado
- persistencia de params batch en `BacktestRun.config_snapshot` (o campo equivalente)

## Compatibilidad con LangGraph/LangChain
Dado que el grafo actual se invoca como `graph.invoke(...)`:
- En `sync`: se mantiene.
- En `batch`: evitar intentar “batch” dentro de LangGraph.
  - En su lugar, el executor debe operar en un nivel donde **se puedan construir requests deterministas** por unidad de análisis.

Esto puede requerir (decisión de implementación) una de estas estrategias:
1) Extraer una “plantilla” de request por step/agent (mínimo viable para batch).
2) Definir un modo batch a nivel “strategy evaluation” (una request = una evaluación completa), si el pipeline lo permite.

La selección final queda en PL como tarea de spike, con criterio de minimizar cambios.

## Riesgos
- Equivalencia de outputs entre modo sync (LangGraph) y modo batch (requests directas) puede no ser 1:1.
- Latencia: batches pueden tardar minutos; el backtest debe tolerarlo.
- Límites de tamaño de archivos y modelo único por batch (OpenAI) obligan a segmentar.

## Decisiones pendientes (si aparecen trade-offs)
Si durante el spike se detectan alternativas con impacto, registrar ADR en `docs/04_decisions/QuantAgent-um8-DC-*.md`.
