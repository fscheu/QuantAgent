# Hallazgos browser-driven

## Resumen ejecutivo

La validación manual post-deploy contra `http://127.0.0.1:8501` quedó en estado `PARTIAL`.

Se confirmó que:
- el runtime Streamlit está vivo en `8501`
- la UI carga en browser real
- no hubo errores de consola JavaScript
- la superficie de paper trading existe y muestra señales de degradación operativa

No se pudo confirmar el runtime paper end-to-end porque la UI expuso ausencia de heartbeat del scheduler y errores de esquema/tablas al consultar `trades`, `orders` y `logs`.

## Evidencia puntual

### 1) Health/liveness del deploy

Observado por terminal antes del browser:
- `GET http://127.0.0.1:8501/_stcore/health` -> `HTTP/1.1 200 OK`
- body -> `ok`

Interpretación:
- el deploy de Streamlit está arriba
- el cutover a `8501` sigue vigente

### 2) Carga inicial de UI

La home de Streamlit cargó con:
- título visible: `QuantAgent – Streamlit MVP`
- selector de entorno visible con valor `paper`
- tabs visibles: `Dashboard`, `Paper Trading`, `Configuration`, `Analyses`, `Backtesting`, `Replay`, `Orders & Positions`, `Logs`

También apareció un mensaje global:
- `Set DATABASE_URL and start PostgreSQL via docker-compose for full functionality.`

Interpretación:
- la app está publicada y navegable
- ya desde el primer frame hay una señal de entorno incompleto para funcionalidad total

### 3) Dashboard

Hallazgos observables:
- métrica `Open Positions` visible en `0`
- métrica `Open Orders` visible en `0`
- estado textual: `Status: 🔴 Stopped`
- línea adicional: `Last run: Never | Errors: -`
- alerta de consulta fallida:
  - `No trades or error reading trades: (psycopg2.errors.UndefinedTable) relation "trades" does not exist`

Interpretación:
- la UI no confirma scheduler sano
- el dashboard cae en estado detenido/no inicializado
- no hay base suficiente para validar histórico de trades ni diferenciar vacío sano vs falla de persistencia

### 4) Paper Trading

Hallazgos observables:
- heading: `📊 Paper Trading Scheduler`
- alerta: `No scheduler heartbeat found`
- mensaje explicativo: `The scheduler may not be running, or no cycles have completed yet for environment: paper`
- instrucción visible: `python apps/paper_trading.py`

Interpretación:
- no hay heartbeat observable al momento del run
- la UI expone explícitamente estado degradado/no inicializado del runtime paper
- AC1 no pudo validarse en sentido positivo; sólo se observó el caso negativo/degradado

### 5) Orders & Positions

Hallazgos observables:
- heading: `Orders & Positions (paper)`
- alerta de consulta fallida:
  - `No orders/positions or error reading: (psycopg2.errors.UndefinedTable) relation "orders" does not exist`

Interpretación:
- no hay evidencia verificable de órdenes ni posiciones activas persistidas
- AC2 y AC3 quedan bloqueados por falta de tablas/datos observables

### 6) Logs

Hallazgos observables:
- heading: `Logs`
- filtros de entorno, nivel, símbolo y event type presentes en UI
- alerta de consulta fallida:
  - `Error querying logs: (psycopg2.errors.UndefinedTable) relation "logs" does not exist`

Interpretación:
- la superficie de observabilidad existe, pero no quedó operativa por persistencia ausente/incompleta
- no hay evidencia en UI para reconstruir runs recientes del scheduler

### 7) Consola del browser

Resultado:
- `console_messages = []`
- `js_errors = []`

Interpretación:
- no aparecieron errores front-end ni excepciones JS durante la navegación manual
- el problema observable es de datos/runtime backend, no de renderizado cliente

## Evaluación contra acceptance criteria

| AC | Resultado | Base observable |
|---|---|---|
| AC1 — Scheduler observable | FAIL observable / no validado en positivo | La UI mostró `No scheduler heartbeat found`. |
| AC2 — Flujo operativo mínimo LONG/SHORT | BLOCKED | No hay tablas/datos visibles de `orders`, `trades` ni `active positions`. |
| AC3 — HOLD no contamina estado | BLOCKED | No hubo datos ni eventos verificables para comparar. |
| AC4 — Visibilidad UI/servicio con estado vacío graceful | PARTIAL | Hay visibilidad de estado, pero en vez de vacío sano aparecieron errores SQL por tablas ausentes. |
| AC5 — Caída detectable | PASS parcial | El sistema sí expone degradación/no heartbeat de forma explícita en UI. |
| AC6 — Evidencia reproducible | PASS | Este sobre con artifacts deja evidencia, paths y screenshots reproducibles. |

## Clasificación final

`PARTIAL`

## Motivo de la clasificación

Se pudo validar el deploy/UI/browser y parte de la visibilidad de degradación, pero no el runtime paper endurecido de punta a punta. Los bloqueos concretos observados fueron:
- falta de heartbeat del scheduler
- consultas a `trades`, `orders` y `logs` fallando por tablas inexistentes
- imposibilidad de verificar flujo order -> trade -> active position desde la UI

## Screenshots guardados

- `screenshots/dashboard.png`
- `screenshots/paper-trading.png`
