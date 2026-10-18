# QuantAgent-d4cq.76: partición de C11

El ticket C11 (`QuantAgent-d4cq.76`) anticipó en su criterio de aceptación:
*"Diff de 100 líneas o menos; si no entra, se parte en lote/ticket y reclamo"*.
La implementación completa (soporte de `LOOP_LOTE`/`--lote`, `LOOP_TICKET`/`--ticket`, verificación de elegibilidad, reclamo `CLAIM`/`RELEASE` con ventana de 6 horas, prompt y wrapper) sumó 156 líneas agregadas+borradas medidas con `scripts/loop/diff_size.sh origin/lote/cartera`, excediendo el objetivo de 100 líneas.
El trabajo preliminar completo con ambas partes está resguardado en `loop-wip/QuantAgent-d4cq.76`.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-d4cq.76.1 | Loop: lote y ticket por variable o argumento (`LOOP_LOTE`, `LOOP_TICKET`, `--lote`, `--ticket`) | Test: con dos lotes borrador abiertos, sin LOOP_LOTE el gate espera; con LOOP_LOTE=lote/x toma ese; test: con --ticket no elegible termina sin entrega y dice por qué; diff <=100 líneas; CI verde | 70–90 | leer |
| QuantAgent-d4cq.76.2 | Loop: reclamo por ticket (`CLAIM`/`RELEASE` y ventana de 6 horas) | Test: dos CLAIM del mismo ticket en menos de 6 horas, el segundo no lanza; con RELEASE o pasadas 6 horas, sí; diff <=100 líneas; CI verde | 60–80 | leer |

Orden: .76.1 → .76.2. La cola de §4 de PLAN-CONTINUACION.md reemplaza la fila del padre por los dos hijos.
El padre queda `in_progress` hasta cerrar los dos hijos.
Cada hijo respeta el límite estricto de ≤100 líneas agregadas+borradas.
