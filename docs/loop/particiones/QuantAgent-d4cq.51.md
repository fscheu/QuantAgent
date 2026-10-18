# QuantAgent-d4cq.51: partición de C06

El ticket C06 (`QuantAgent-d4cq.51`) anticipó en su criterio de aceptación:
*"Diff de 100 líneas o menos; si no entra, se parte en combinación y alarma"*.
La implementación completa (combinación de equities por celda con pesos y combinar, V30 sobre curva combinada, control por celda, alarma de cartera, integración en CLI `correr` retirando guarda de falta C06, registro D7 en `intentos.csv` y tests exhaustivos) superaría las 250 líneas agregadas+borradas (~90 en código de producción y ~160 en tests), excediendo el objetivo de 100 líneas.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-d4cq.51.1 | Combinación de etapa 2 y V30 de cartera | Test con compare falso (equities fijas por celda): V30 se evalúa sobre la curva combinada; una celda que no coincide -> CORTA aunque la cartera pase; formato de una pierna da el mismo resultado; diff <=100 líneas; CI verde | 60–80 | leer |
| QuantAgent-d4cq.51.2 | Alarma de cartera, CLI etapa 2 y registro D7 | Test de la alarma de cartera (dispara y no dispara); test de 2 piernas x 3 activos agrega exactamente 1 fila en intentos.csv para etapa 2; diff <=100 líneas; CI verde | 70–90 | leer |

Orden: .51.1 → .51.2. La cola de §4 de PLAN-CONTINUACION.md reemplaza la fila del padre.
El padre queda `in_progress` hasta cerrar los dos hijos; C07 (`QuantAgent-d4cq.52`) sigue dependiendo del padre.
Cada hijo respeta el límite estricto de ≤100 líneas agregadas+borradas.
