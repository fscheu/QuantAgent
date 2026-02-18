# QuantAgent-69d: Tracking de tokens y tiempo de ejecución - Acceptance Criteria

## AC-1: Persistencia por llamada LLM
**Given** una ejecución que realiza al menos 1 llamada LLM
**When** la llamada finaliza (success)
**Then** existe un registro persistido con:
- `provider`, `model`, `operation`
- `duration_ms > 0`
- `input_tokens` y `output_tokens` presentes si el provider los reporta (si no, NULL)

## AC-2: Registro en caso de error
**Given** una llamada LLM que falla (timeout / auth / provider error)
**When** se captura el error
**Then** se persiste un registro con:
- `duration_ms > 0`
- `extra_data.status = "error"` (o equivalente)
- tokens NULL si no están disponibles

## AC-3: Asociación a backtest
**Given** un backtest con `backtest_run_id = X`
**When** se ejecutan llamadas LLM durante el backtest
**Then** cada registro persistido tiene `backtest_run_id = X`

## AC-4: Agregación por backtest
**Given** un backtest con múltiples llamadas LLM
**When** se consultan agregados por `backtest_run_id`
**Then** el resultado incluye:
- `calls` (>=1)
- suma de tokens (por tipo)
- suma/avg de `duration_ms`
- posibilidad de breakdown por `operation`

## AC-5: Agregación por sesión (thread_id)
**Given** una ejecución paper/prod que usa `thread_id = T`
**When** se consultan agregados por `thread_id = T`
**Then** el resultado contiene métricas agregadas sólo de esa sesión

## AC-6: Exposición para análisis
**Given** un backtest existente con métricas
**When** se solicita la vista/consulta de métricas (servicio interno o UI)
**Then** se muestran al menos:
- total tokens (input/output/total si existe)
- duración total
- breakdown por `operation`

## Negative cases
### NC-1: No contaminación entre backtests
**Given** dos backtests A (`run_id=1`) y B (`run_id=2`)
**When** se consultan agregados por `run_id=1`
**Then** no incluye llamadas con `backtest_run_id=2`

### NC-2: Provider sin usage
**Given** un provider que no retorna tokens
**When** se persiste el registro
**Then** tokens = NULL y `duration_ms` sigue presente
