# Partición de QuantAgent-av3i (V31a)

El comando completo requiere unas 185 líneas de código y tests: grilla y
validaciones (~35), CLI/candado/CSV (~75), tests unitarios y de integración (~75).
Supera el objetivo de 100 antes de escribir código. No se implementó sweep.

| Hijo | Objetivo | Estimación | Aceptación binaria | Revisión |
| --- | --- | --- | --- | --- |
| QuantAgent-av3i.1 | Cargar y validar grilla | 75 líneas | RSI 2x2 devuelve cuatro combinaciones; parámetros desconocidos y >50 se rechazan | leer |
| QuantAgent-av3i.2 | CLI, ejecución aislada y CSV con candado | 95 líneas | Entradas inválidas salen 1 sin CSV; no hay flags de apertura | probar |
| QuantAgent-av3i.3 | Integración sintética y comprobación VM | 90 líneas | Cuatro filas; defaults = correr_par; RSI/SPY 14: 187/-9141.21, 7: 325/-14560.44 | probar |

Las estimaciones por hijo incluyen margen para integración y correcciones.
Orden: .1 → .2 → .3, con dependencias en Beads y filas L7 en
[la cola](../../../PLAN-CONTINUACION.md). Cada entrega corre suite y smoke.
Los contratos completos y comandos de aceptación están en `bd show` de cada hijo.
Los tests usan el registro y el motor reales; no sustituyen correr_par por mocks.

El padre queda in_progress hasta cerrar los tres hijos. V31b sigue dependiendo
del padre: no se desbloquea al aprobar esta partición ni al cerrar solo .1 o .2.
No se cambian motor, estrategias, intentos ni decisiones pendientes de V30.
Ninguna comprobación necesita abrir 2019–2022 ni la reserva.

Comando de revisión: `test -f docs/loop/particiones/QuantAgent-av3i.md && bd show QuantAgent-av3i.1 && bd show QuantAgent-av3i.2 && bd show QuantAgent-av3i.3`.
