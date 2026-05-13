# QuantAgent-sft — Acceptance criteria: paper runtime hardening

## AC1 — Scheduler observable
**Given** el entorno QA de paper trading está configurado
**When** el scheduler arranca
**Then** existe un heartbeat reciente y verificable sin inspección manual del proceso

## AC2 — Flujo operativo mínimo
**Given** una decisión LONG o SHORT válida en paper mode
**When** `OrderManager` la ejecuta
**Then** quedan registros consistentes de orden/trade y el estado de posición activa refleja el resultado esperado

## AC3 — HOLD no contamina estado
**Given** una decisión HOLD
**When** el flujo de ejecución corre
**Then** no se crea una orden/trade espuria ni se altera indebidamente la posición activa

## AC4 — Visibilidad en UI/servicio
**Given** la UI de paper trading consulta el estado operativo
**When** no hay trades pero el scheduler está sano
**Then** la UI muestra estado vacío/graceful sin reportar falsamente una caída del runtime

## AC5 — Caída detectable
**Given** el scheduler no dejó heartbeat dentro de la ventana definida por el ticket
**When** se consulta el estado operativo
**Then** el sistema expone esa condición como runtime degradado o no saludable

## AC6 — Evidencia de piloto controlado
**Given** la implementación está completa
**When** tester o tech lead ejecutan la verificación M2
**Then** existen comandos, artefactos y oráculos suficientes para repetir la validación en QA sin depender de conocimiento implícito
