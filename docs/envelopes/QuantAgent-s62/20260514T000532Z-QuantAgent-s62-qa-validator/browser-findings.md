# Browser findings — QuantAgent-s62

## Metodología

Validación manual sobre browser real conectado al target ya abierto en `http://127.0.0.1:8501/`. Se recorrieron las pestañas `Dashboard`, `Paper Trading` y `Logs`, se inspeccionó texto renderizado, se revisó consola JS y se capturaron screenshots.

## Estado general de la app

- URL: `http://127.0.0.1:8501/`
- Título del documento: `QuantAgent UI (MVP)`
- Health endpoint: `ok`
- Banner persistente visible en UI: `Set DATABASE_URL and start PostgreSQL via docker-compose for full functionality.`
- Selector global de environment: `paper`
- Console browser: sin mensajes y sin `js_errors`

## Evidencia por pestaña

### 1) Dashboard

Texto relevante observado:

- `Portfolio Value` => `-`
- `Daily P&L` => `-`
- `Win Rate` => `-`
- `Open Positions` => `0`
- `Open Orders` => `0`
- `Recent Trades` => `No trades or error reading trades: (psycopg2.errors.UndefinedTable) relation "trades" does not exist ...`
- `Scheduler Status`
- `Status: 🔴 Stopped`
- `Last run: Never | Errors: -`
- `Full detail: Paper Trading tab`

Interpretación QA:

- Positivo: el scheduler status ya no muestra placeholder MVP; hay wiring real de estado.
- Negativo: la sección de trades degrada con error SQL crudo, no con mensaje amigable de `No data` / `DB not available`.

### 2) Paper Trading

Texto relevante observado:

- `📊 Paper Trading Scheduler`
- `No scheduler heartbeat found`
- `The scheduler may not be running, or no cycles have completed yet for environment: paper`
- `How to start the scheduler:`
- `python apps/paper_trading.py`

Interpretación QA:

- Positivo: existe degradación explícita del heartbeat/scheduler.
- Negativo crítico: no se observaron, ni siquiera en estado vacío, las secciones documentadas para `Positions & Orders`, `PnL Summary` y `LLM Cost & Latency (last 24h)`.
- Consecuencia: no queda validada la ruta de no-data de AC2/AC3; más aún, visualmente parece no renderizarse.

### 3) Logs

Controles visibles:

- Selectbox `Environment`
- Opciones visibles: `all`, `paper`, `backtest`
- Multi-select `Log Level`
- `Symbol (contains)`
- `Event Type (contains)`
- `Hours Back`

Evidencia funcional del filtro de environment:

- Con `Environment = all`, el SQL visible en el mensaje de error no incluía filtro por `logs.environment`.
- Luego de cambiar a `Environment = paper`, el SQL visible pasó a incluir:
  - `AND logs.environment = %(environment_1)s`
  - parámetro `environment_1 = 'paper'`

Texto relevante observado en error:

- `Error querying logs: (psycopg2.errors.UndefinedTable) relation "logs" does not exist ...`

Interpretación QA:

- Positivo: el wiring del filtro de environment está operativo y observable.
- Negativo: la degradación sigue mostrando SQL crudo por tabla faltante.

### 4) Orders & Positions (hallazgo colateral)

Aunque no era el foco principal del ticket, al abrir la pestaña apareció:

- `No orders/positions or error reading: (psycopg2.errors.UndefinedTable) relation "orders" does not exist ...`

Esto refuerza que el entorno activo no tiene esquema utilizable para validación positiva y que la degradación actual expone internals SQL.

## Consola del browser

Resultado de `browser_console` durante toda la corrida:

- `console_messages`: `[]`
- `js_errors`: `[]`
- Conclusión: no se observaron errores frontend/JS; los problemas vistos son del lado de datos/render backend.

## Capturas guardadas

- `screenshots/dashboard.png`
- `screenshots/paper-trading.png`
- `screenshots/logs.png`

## Conclusión operativa

La app está viva y navegable, pero el entorno post-deploy no permite validar escenarios positivos con datos reales. Además, en escenarios vacíos/rotos la degradación observable es inconsistente: parte del scheduler degrada bien, pero otras superficies muestran SQL crudo o directamente no renderizan secciones esperadas.
