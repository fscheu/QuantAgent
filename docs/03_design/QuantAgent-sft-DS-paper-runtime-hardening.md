# QuantAgent-sft — Design: paper runtime hardening

## Design level
STANDARD

## Current state
El camino técnico principal ya existe:
- `quantagent/trading/scheduler.py` coordina ciclo y heartbeat
- `quantagent/trading/order_manager.py` ejecuta decisiones y persiste órdenes/trades
- `quantagent/trading/position_monitor.py` mantiene posiciones activas
- `apps/streamlit/` ya expone la vista de paper trading y tiene validación PoC (`QuantAgent-vje`)

La brecha no es de arquitectura nueva sino de endurecimiento operativo y evidencia.

## Proposed change
Completar el milestone usando los componentes actuales y reforzando cuatro seams:

1. **Boot/runtime seam**
   - un único path documentado de arranque para QA
   - configuración explícita de entorno paper

2. **Heartbeat seam**
   - `SchedulerHeartbeat` debe reflejar ejecución reciente y utilizable como oracle operativo

3. **Execution seam**
   - decisiones procesadas por `OrderManager` deben dejar estado consistente en `Order` / `Trade` / `ActivePosition`

4. **Visibility seam**
   - UI o servicios existentes deben poder distinguir:
     - runtime sano sin actividad
     - runtime caído o no inicializado

## Affected components
- `quantagent/trading/scheduler.py`
- `quantagent/trading/order_manager.py`
- `quantagent/trading/position_monitor.py`
- `apps/streamlit/views/paper_trading.py` y/o servicios asociados
- tests del scheduler / order flow / UI services

## Technical decisions
- Reusar `SchedulerHeartbeat` como señal primaria de liveness del scheduler.
- Reusar persistencia actual de órdenes/trades/positions; no introducir un estado paralelo.
- Reusar QA Streamlit como superficie de observación mínima para M2.
- Tratar `QuantAgent-69d` como mejora de telemetry complementaria, no como prerequisito para el runtime base.

## Validation shape
La implementación queda lista cuando pueda demostrarse:
1. arranque del runtime paper en QA,
2. heartbeat reciente,
3. consistencia mínima del flujo orden→trade→active position,
4. visibilidad de estado en la UI o servicio de UI.
