# QuantAgent — Milestone Tracker

## Milestone actual
- Milestone: M2 — Validación con datos reales
- Progreso: lote 1 de 7 abierto el 2026-10-09 (`lote/datos-reales-diarios`). Plan y cola: `docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md` y `PLAN-CONTINUACION.md` §4.
- M1 cumplido el 2026-10-09 (PR #52): tres estrategias reproducibles sobre fixtures, métricas auditadas y engine igual a `backtesting.py` trade por trade.
- Fecha objetivo: sin fecha. Ritmo: lotes de 5–10 entregas revisadas por el PM; Fede revisa un PR por lote.
- Estado: on track

## Milestones

| Milestone | Nombre | Criterio de completitud | Fecha objetivo | Estado |
|---|---|---|---|---|
| M1 | Backtesting estable | Suite de backtesting corre sin bugs para 3 estrategias distintas, resultados reproducibles | 2026-10-09 | cumplido |
| M2 | Validación con datos reales | Comparación de N estrategias sobre datos reales con costos, contra comprar y mantener, con veredicto fuera de muestra (`docs/02_planning/QuantAgent-lp1-PL-m2-validacion-datos-reales.md`) | TBD | TBD |
| M3 | Broker Alpaca en paper | Integración con Alpaca funcionando contra la cuenta paper | TBD | TBD |
| M4 | Paper trading estable | La estrategia validada corre en paper trading 2 semanas sin intervención manual | TBD | TBD |
| M5 | Motor top-down | Macro y momentum eligen sectores, el screener elige acciones y los stops viven en el broker, todo en paper (`docs/02_planning/QuantAgent-M5-PL-motor-top-down.md`) | TBD | TBD |

## Después de M5
- Decidido por Fede el 2026-10-28 (G1): M5 es el motor top-down. El dinero real ya no es un hito con número: es la salida de M-ciclos + M-cuenta, con autorización explícita de Fede y ganándole a comprar y mantener SPY con USD 2.500.
- Los hitos siguientes se llaman por tema (M-manual, M-datos, M-web, M-ciclos, M-cuenta) y viven en [`PLAN-POST-M5.md`](./PLAN-POST-M5.md).
- Esta tabla sigue desactualizada en M2–M4 (el progreso real está en `PLAN-CONTINUACION.md`); no se reescribió aquí.

## Notas
- Editar “Milestone actual” cuando se cambie de foco.
- (Opcional) Agregar una lista de tickets por milestone para que el briefing pueda calcular X/Y.
