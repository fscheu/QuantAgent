# Partición de QuantAgent-ev7d

El cambio estimado supera el objetivo de 100 líneas porque combina el flujo de cierre en `OrderManager` con
las rutas de salida y sus pruebas de integración en `TradingScheduler`.

| Hijo | Objetivo | Criterio binario | Revisión |
|---|---|---|---|
| [QuantAgent-ev7d.1](../../../PLAN-CONTINUACION.md#q13a) | Agregar cierre por símbolo y compartir validación/registro con `close_trade`. | Ambas funciones cierran la cantidad prevista y registran la orden; pruebas dirigidas verdes. | leer |
| [QuantAgent-ev7d.2](../../../PLAN-CONTINUACION.md#q13b) | Enrutar salidas del scheduler a `close_trade` o `close_symbol` y cubrir los tres escenarios. | Cartera plana tras salida larga, corta y sin `trade_id`; M1 sin cambios y oráculos verdes. | probar |

`ev7d.2` depende de `ev7d.1`. Cada hijo tiene alcance estimado de hasta 100 líneas.
