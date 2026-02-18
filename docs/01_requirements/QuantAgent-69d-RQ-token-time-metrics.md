# QuantAgent-69d: Tracking de tokens y tiempo de ejecución - Requirements

## Objective
Agregar tracking de uso (tokens) y tiempos de ejecución para controlar y optimizar costos del agente.

## Scope
### In-scope
1. **Métricas por llamada LLM** (cada request al provider):
   - `input_tokens` (prompt / input)
   - `output_tokens` (completion / output)
   - `total_tokens` (si está disponible)
   - `duration_ms`
   - Identificadores: `provider`, `model`, `operation` (nombre de nodo/acción), `environment` (backtest/paper/prod), `symbol` (si aplica), `thread_id` y/o `checkpoint_id` (si aplica)
2. **Métricas agregadas**:
   - Agregación **por sesión** (definida por `thread_id`) para ejecuciones tipo paper/prod
   - Agregación **por backtest** (definida por `backtest_run_id`) para backtesting
   - Agregación mínima: sumas de tokens + sumas/promedios/p95 de `duration_ms`, agrupable por `operation` y por `provider/model`
3. **Persistencia** en un storage consultable (DB preferido si está disponible en la ejecución).
4. **Exposición para análisis**:
   - Query API interna (funciones/servicio) para obtener: (a) métricas por llamada y (b) agregados por sesión/backtest.
   - Visualización mínima en UI (si existe vista de backtests): mostrar tokens y tiempos agregados por backtest.

### Out-of-scope (por ahora)
- Cálculo de costo en USD (depende de precios por modelo, cambios frecuentes)
- Métricas de infraestructura (CPU, RAM)
- Métricas de tools no-LLM, salvo que ya existan hooks/callbacks directos

## Constraints
- **Overhead bajo**: el tracking no debe agregar más de ~5% de latencia promedio por llamada (target).
- **No romper providers**: debe funcionar con `openai`, `anthropic`, `qwen`, `azure`.
- **No depender de chat history**: toda medición debe derivar de la ejecución real.

## Data Model (conceptual)
### LLMCallMetric
Registro atómico por llamada:
- Identidad: `timestamp`, `provider`, `model`, `operation`
- Uso: `input_tokens`, `output_tokens`, `total_tokens`
- Tiempo: `duration_ms`
- Contexto: `environment`, `symbol`, `thread_id`, `checkpoint_id`, `backtest_run_id`
- `extra_data` (JSON): ids/campos que el provider exponga (por ejemplo, `response_id`).

### Aggregated Metrics
- `SessionMetrics` (clave: `thread_id`)
- `BacktestMetrics` (clave: `backtest_run_id`)

## Definition of Done
- Se persiste **al menos** 1 registro por cada llamada LLM con tokens + duration.
- Se puede consultar agregados por `thread_id` y por `backtest_run_id`.
- En backtesting, se pueden comparar backtests por: tokens totales, duración total y breakdown por `operation`.

## Edge Cases
- Provider no retorna tokens: registrar `input_tokens/output_tokens/total_tokens = NULL` y mantener `duration_ms`.
- Errores de llamada LLM: se registra el intento con `event/status=error` (en `extra_data`) y `duration_ms` igualmente.
- Ejecución sin DB: el sistema debe poder degradar a logging/archivo (si ya existe configuración), pero esto queda como fallback opcional en implementación.
