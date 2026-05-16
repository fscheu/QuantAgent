# Docs check — QuantAgent-s62

## Resumen

Se contrastaron RQ/DS/AC/manuales contra lo observable en la UI Streamlit levantada en `http://127.0.0.1:8501/`.

## Hallazgos por criterio

| AC | Expectativa documental | Evidencia observable | Estado |
|---|---|---|---|
| AC1 | Dashboard debe reemplazar el placeholder de scheduler por estado real; con ausencia de heartbeat debe mostrar `No data` o `Stopped`. | En Dashboard se vio `Scheduler Status` con `Status: 🔴 Stopped` y `Last run: Never | Errors: -`. No aparece el texto placeholder `unknown (MVP placeholder)`. | PARTIAL |
| AC2 | Paper Trading debe mostrar scheduler + posiciones + órdenes + resumen PnL; sin datos debe mostrar mensajes explícitos por sección. | En Paper Trading sólo se vio el bloque de scheduler (`No scheduler heartbeat found`) y comando `python apps/paper_trading.py`. No se renderizaron secciones visibles de posiciones, órdenes ni PnL en estado vacío. | FAIL |
| AC3 | Paper Trading debe mostrar bloque `LLM Cost & Latency`; sin telemetry debe mostrar `No LLM telemetry data found for this environment.` | No se observó ninguna sección LLM en Paper Trading. Tampoco apareció el mensaje explícito de no telemetry. | FAIL |
| AC4 | Logs debe ofrecer filtro de environment (`all/paper/backtest`) y aplicar filtro al query. | Se observó selectbox `Environment` con opciones `all`, `paper`, `backtest`. En `all` el SQL visible no incluía filtro por environment; al cambiar a `paper`, el SQL visible agregó `AND logs.environment = %(environment_1)s`. | PASS |
| AC5 | Degradación explícita y usable ante DB no disponible o sin datos, sin romper la UI. | No hubo crash total ni errores JS en consola, pero la UI expuso errores SQL crudos (`UndefinedTable`) en Dashboard, Logs y Orders & Positions; además Paper Trading no mostró todas las secciones degradadas esperadas. | FAIL |

## Chequeo contra requisitos y diseño

- FR1/DD4: observado y consistente en lo mínimo; el dashboard ya no usa placeholder MVP para scheduler.
- FR2/DD5: no se pudo validar el bloque integrado completo; en estado sin datos no se observó la degradación por sección documentada.
- FR3/DD5: no visible en browser; no aceptable para la ruta vacía documentada.
- FR4/DD6: visible y funcional a nivel de wiring UI/query.
- FR5/Degradation Contract: incumplido en esta instancia observable por exposición de errores SQL crudos y ausencia de bloques degradados esperados.

## Conclusión documental

La implementación observable cumple parcialmente la intención del issue, pero la evidencia post-deploy no alcanza para aceptar `QuantAgent-s62` en este entorno. Hay una combinación de falta de datos/DB válida para escenarios positivos y, además, degradación incompleta o poco amigable en escenarios vacíos/rotos.
