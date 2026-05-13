# QuantAgent-sft — Planning: paper runtime hardening

## Objective
Llevar el runtime paper de “componentes presentes” a “operación verificable de punta a punta” para M2.

## Dependencies
- `QuantAgent-s62` para el frente QA/UI de cutover Streamlit.
- `QuantAgent-339` para validación browser post-deploy más robusta.
- `QuantAgent-69d` como mejora complementaria de telemetry, no bloqueante para el runtime base.

## Task breakdown

### 1. Runtime preflight
- Confirmar el path canónico de arranque paper en QA.
- Verificar settings/env mínimos que distinguen paper de otros modos.
- Delimitar la señal de liveness que usará aceptación (`SchedulerHeartbeat` y/o servicio UI asociado).

### 2. Scheduler + heartbeat hardening
- Ajustar el scheduler para que el heartbeat sea consistente y testeable.
- Cubrir casos de “arrancó pero no está procesando” vs “no hay actividad de mercado”.

### 3. Execution consistency
- Validar y completar el flujo `OrderManager` → persistencia → `ActivePosition`.
- Cubrir explícitamente LONG/SHORT/HOLD y consistencia de estado resultante.

### 4. UI/service visibility
- Asegurar que la superficie de paper trading consuma la señal operativa correcta.
- Diferenciar estado vacío de estado degradado.

### 5. Verification
- Tests focalizados de scheduler/order flow/UI service.
- Smoke manual en QA con evidencia reutilizable para el milestone M2.

## Risks
- Heartbeat presente pero semánticamente débil para distinguir salud real.
- Tests cubren DB/state pero no la lectura real desde la UI.
- Dependencia parcial de datos de mercado puede volver frágil la prueba manual.

## Recommended routing
1. `autodev-implementer`
2. `autodev-tester`
3. Tech Lead integration sólo si hay evidencia de runtime + UI/estado operativo suficiente
