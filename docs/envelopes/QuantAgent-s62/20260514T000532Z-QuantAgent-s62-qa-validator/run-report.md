# Run report — 20260514T000532Z-QuantAgent-s62-qa-validator

**Issue:** QuantAgent-s62  
**Fase:** qa-validator  
**Resultado:** PARTIAL  
**Decisión de aceptación:** NO-GO  
**Branch:** `main`  
**Commit:** `76dfa0bc84cc02938984190e1bc783d32abe488f`

## Qué hice

- Leí RQ, DS, AC y manuales operativos vinculados al issue.
- Verifiqué precondiciones del deploy local:
  - `http://127.0.0.1:8501/_stcore/health` => `ok`
  - el merge commit requerido `089a1c559ce77312022ab89072f31ffbbbb54b81` está contenido en `HEAD`
- Navegué la app real con browser sobre `Dashboard`, `Paper Trading`, `Logs` y también observé `Orders & Positions` como hallazgo colateral.
- Revisé la consola del browser en múltiples puntos de la corrida.
- Guardé evidencia durable en esta carpeta, incluyendo capturas.

## Hallazgos principales

### 1. Dashboard

- `Scheduler Status` ya está cableado a estado real y no al placeholder MVP.
- Evidencia visible:
  - `Status: 🔴 Stopped`
  - `Last run: Never | Errors: -`
- Problema observado: `Recent Trades` expone error SQL crudo por tabla faltante (`trades`).

### 2. Paper Trading

- Sólo se observó el bloque del scheduler:
  - `No scheduler heartbeat found`
  - `The scheduler may not be running...`
  - `python apps/paper_trading.py`
- No se observaron las secciones documentadas de:
  - positions/orders,
  - PnL summary,
  - LLM Cost & Latency.
- Esto rompe la expectativa de degradación explícita para estado vacío.

### 3. Logs

- El filtro `Environment` está presente y visible con opciones `all`, `paper`, `backtest`.
- Evidencia funcional:
  - con `all`, el SQL visible no agregaba filtro por environment;
  - con `paper`, el SQL visible agregó `AND logs.environment = %(environment_1)s`.
- Problema observado: la vista expone error SQL crudo por tabla faltante (`logs`).

### 4. Consola del browser

- Sin `console_messages`
- Sin `js_errors`
- La UI no crasheó a nivel frontend; los problemas visibles son de datos/render backend.

## Evaluación por acceptance criteria

| AC | Estado | Nota |
|---|---|---|
| AC1 | PARTIAL | Se validó la desaparición del placeholder y el estado `Stopped`; no hubo heartbeat real para validar escenario positivo. |
| AC2 | FAIL | No se renderizaron secciones visibles de posiciones/órdenes/PnL en Paper Trading vacío. |
| AC3 | FAIL | No se observó bloque LLM ni mensaje de no telemetry. |
| AC4 | PASS | El filtro de environment de Logs existe y modifica el query observable. |
| AC5 | FAIL | Hay degradación parcial, pero también errores SQL crudos y ausencia de bloques degradados esperados. |

## Contexto de entorno que afectó la validación

- La UI muestra: `Set DATABASE_URL and start PostgreSQL via docker-compose for full functionality.`
- En shell, el `.env` local apunta a una DB PostgreSQL inexistente (`database "quantagent" does not exist`).
- En la UI se observaron además tablas faltantes (`orders`, `trades`, `logs`).
- Resultado: no hubo base de datos/esquema/seed confiable para validar escenarios positivos con datos reales.

## Conclusión

La corrida manual post-deploy fue ejecutada y dejó evidencia durable, pero el resultado no es aceptable para cierre funcional del issue en este entorno. Hay avance visible en Dashboard y Logs, pero Paper Trading no exhibe el comportamiento documentado en vacío y la degradación general todavía expone errores SQL crudos.

## Próximo paso recomendado

1. Corregir la degradación vacía/DB rota en `Paper Trading`, `Dashboard` y `Logs` para no exponer SQL crudo.
2. Levantar una DB con schema correcto y seed mínimo (`SchedulerHeartbeat`, `Order`, `Trade`, `Log`).
3. Repetir esta validación manual browser-driven sobre el mismo target o uno equivalente.
