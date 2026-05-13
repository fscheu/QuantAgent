# QuantAgent-sft — Requirements: paper runtime hardening

## Objective
Cerrar la brecha entre el paper trading “implementado en componentes” y el paper trading “operable de punta a punta” en entorno QA/no productivo.

## Context
El repo ya tiene piezas core presentes:
- `TradingScheduler` con heartbeat y `PositionMonitor`
- `OrderManager` / `PaperBroker` / `RiskManager`
- modelos `Order`, `Trade`, `Signal`, `ActivePosition`, `SchedulerHeartbeat`
- UI Streamlit con tab de paper trading validado en PoC (`QuantAgent-vje`)

El ticket debe transformar ese baseline en un runtime estable y verificable para M2.

## Scope
1. Asegurar que el runtime de paper trading pueda arrancar y mantenerse estable en QA.
2. Validar el camino operativo mínimo:
   - scheduler activo
   - heartbeat observable
   - decisiones LONG/SHORT/HOLD procesadas por `OrderManager`
   - posiciones activas reflejadas por `PositionMonitor`
3. Hacer observable el estado operativo mínimo desde logs, DB o UI ya existente.
4. Dejar evidencia reproducible para piloto controlado interno.

## Out of scope
- Broker real / Alpaca / live trading.
- Nuevas estrategias o cambios de alpha.
- Replanteo arquitectónico del scheduler.
- Observabilidad “enterprise” (Prometheus, tracing distribuido, alerting externo).
- Automatización de QA browser completa; eso pertenece a `QuantAgent-339`.

## Constraints
- Reusar los componentes actuales; cambios mínimos.
- No introducir nuevas tablas si las existentes (`logs`, `SchedulerHeartbeat`, órdenes/trades/positions) alcanzan.
- Mantener el entorno QA/no productivo como único target del milestone M2.
- Los checks deben distinguir ausencia de datos vs falla operativa real.

## Edge cases
- Scheduler corre pero no deja heartbeat reciente.
- UI muestra vacío por falta de actividad, pero el runtime está sano.
- Se generan órdenes pero no quedan consistentes `Trade` / `ActivePosition`.
- Reinicio del scheduler deja estado ambiguo sobre posiciones abiertas.

## Definition of done
- Existe una forma reproducible de arrancar/verificar el runtime paper en QA.
- El scheduler deja evidencia observable de ejecución reciente.
- El flujo orden/trade/active-position queda verificable con tests y/o chequeos manuales concretos.
- La documentación de operación mínima para M2 queda lista para implementer/tester.
