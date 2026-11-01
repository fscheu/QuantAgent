# QuantAgent-fnpr.6.6.4: diario sobre acciones candidatas

El diario lee `m5-candidatos-semanales.csv` junto al CSV de régimen. Selecciona la
última fecha no posterior a la sesión NYSE de decisión y toma símbolos con rank
no nulo, sin duplicados. No mezcla semanas ni incorpora filas futuras o ETFs de
país. Sin archivo, con cero bytes o sólo cabecera conserva la salida histórica
de ETFs, verificada contra órdenes e intents capturados antes del cambio.

El panel incorpora candidatos y posiciones PAPER activas, aunque hayan salido
de selección. Sólo los habilitados y candidatos reciben entradas. El seguimiento
de posiciones precede al bloqueo de entradas por risk-off o régimen sin fila
vigente. Un error del proveedor para una acción conserva el stop previo y deja
motivo en logs; precio ausente, no positivo o sin cierre de decisión e historia
insuficiente también se descartan con motivo, sin frenar los ETFs.

`correr_diario(es_fraccionable=...)` permite inyectar la consulta. Por defecto
usa `broker.es_fraccionable`, implementado en `AlpacaBroker` mediante
`client.get_asset(simbolo).fractionable`; así queda conectado al broker que ya
construye `m5-correr`, sin cambiar el comando ni sus opciones. Si el broker no
expone la consulta, las acciones usan enteros. Un error al consultar el activo
descarta la acción con motivo. Los ETFs conservan su sizing anterior.

Se usa el sizer existente sin cambiar riesgo, tope ni heat del bucket. Las
acciones fraccionables se truncan a ocho decimales antes de validar riesgo y
persistir; las demás usan el redondeo entero existente. Cantidad cero persiste
un intent `rechazado_riesgo` con `tamano_cero`. Las entradas de acciones se
adaptan a stop, incluido B, al precio de entrada de la señal y con stop de venta
protector por igual cantidad. Los arquetipos y las entradas históricas de ETFs
no cambian. Se conserva la deduplicación por sesión/símbolo/estrategia.

Order, fills, ActivePosition y reconciliación ya admitían decimales: no hubo
migración ni cambio de sincronización. El broker conserva su vigencia existente:
stops enteros GTC, fraccionarios DAY. No se validó la aceptación real en Alpaca
ni se implementó renovación de órdenes DAY; no se usó red.

## Validación y entrega

Tests: semana sintética con A/B reales, comparación exacta sin candidatos,
descarte por datos, rechazo entero, consulta inyectada o desde broker,
idempotencia, conservación/actualización de stop al salir de candidatos incluso
en risk-off, fills y reconciliación fraccionaria. Comandos y resultados finales
se entregan en la respuesta de la sesión. No cambió metadata de comandos CLI.
`tests/test_bucket_m5.py` no existe en esta base; su equivalente es
`tests/test_m5_50_bucket_paper.py`. El snapshot real de A requiere datos externos
y DATABASE_URL: la corrida inicial falló por esa configuración. La validación
aislada apunta los snapshots al worktree y omite ese test si no están disponibles.

Suite aislada: **171 passed, 1 skipped, 0 failed**, exit 0, 432,69 segundos.
El omitido es el snapshot real de A, ausente dentro del worktree.
Crear `.cache` antes de repetir desde una entrega limpia: `mkdir -p .cache`.

Comando de suite aislada (xdist requiere DATABASE_URL SQLite explícito):

```bash
DATABASE_URL=sqlite:///$PWD/.cache/tests.db QUANTAGENT_SNAPSHOT_DIR=$PWD/.cache/snapshots PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba MPLCONFIGDIR=$PWD/.cache/matplotlib /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest -n 3 -q tests/test_pipeline_m5.py tests/test_trade_intent.py tests/test_arquetipo_a.py tests/test_arquetipo_b.py tests/test_m5_50_bucket_paper.py tests/test_order_manager*.py tests/test_cli_*.py tests/trading/test_alpaca_broker_orders.py tests/test_position_monitor.py tests/test_m5_54_puesta_en_marcha.py
```

Smoke (exit 0, 9 trades, PnL 561,18):

```bash
PATH=/home/azureuser/repos/projects/QuantAgent/.venv/bin:$PATH PYTHONPATH=. NUMBA_CACHE_DIR=$PWD/.cache/numba MPLCONFIGDIR=$PWD/.cache/matplotlib bash scripts/loop/smoke.sh
```

Entrega sin stage, commits, push, PR ni uso de bd. El issue queda pendiente de
revisión e integración por el PM; no se modificó su estado local.
