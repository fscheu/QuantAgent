# QuantAgent-d4cq.50: partición de C05b

El ticket C05b (`QuantAgent-d4cq.50`) anticipó en su criterio de aceptación:
*"Diff de 100 líneas o menos; si no entra, se parte en cálculo y CLI"*.
La implementación completa (cálculo de cartera, veredicto, integración en CLI, registro D7 en `intentos.csv`, tabla y tests exhaustivos) suma ~310 líneas agregadas+borradas (115 en `quantagent/` y ~190 en `tests/`), excediendo el objetivo de 100 líneas.
El trabajo preliminar está resguardado en `loop-wip/QuantAgent-d4cq.50`.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-d4cq.50.1 | Cálculo de etapa 1 para cartera (`evaluar_cartera` y `veredicto_etapa1`) | 1 pierna/1 activo coincide con Sharpe individual; 24 trades corta por trades; tests de veredicto verdes | 60–80 | leer |
| QuantAgent-d4cq.50.2 | Integración de etapa 1 de cartera en CLI (`filtrar`, `correr`) y registro D7 | 12×5 trades pasa; 2 piernas × 3 activos agrega 1 fila; formato viejo coincide; CI verde | 80–100 | leer |

Orden: .50.1 → .50.2. La cola de §4 de PLAN-CONTINUACION.md reemplaza la fila del padre.
El padre queda `in_progress` hasta cerrar los dos hijos; C06 (`QuantAgent-d4cq.51`) sigue dependiendo del padre.
Cada hijo respeta el límite estricto de ≤100 líneas agregadas+borradas.
