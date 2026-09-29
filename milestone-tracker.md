# QuantAgent — Milestone Tracker

## Milestone actual
- Milestone: M1 — Backtesting estable
- Progreso: 2/14 tickets M1 cerrados (`gg6`, `83e`). Por criterio real (3 estrategias distintas, punta a punta): 1/3 — RSI validado vía `quantagent backtest run` (CLI determinístico, sin API key, D13-D18 de `PLAN-30-DIAS.md`); Triple Screen y 52-week-high aún no producen trades reales (`8u8`, `46h`, ya desbloqueados tras cerrar `83e`). Auditoría de métricas (`hx0` + 4 subtickets), verificación de reproducibilidad (`y8z`) y cross-validación contra `backtesting.py` (`832`, `lcv`) siguen abiertas.
- Fecha objetivo: sin fecha. Ritmo: una entrega revisada por día hábil, en el orden de la cola de `PLAN-CONTINUACION.md` §4 (M1 = T01–T23).
- Estado: on track

## Milestones

| Milestone | Nombre | Criterio de completitud | Fecha objetivo | Estado |
|---|---|---|---|---|
| M1 | Backtesting estable | Suite de backtesting corre sin bugs para 3 estrategias distintas, resultados reproducibles | 2026-10-16 | on track |
| M2 | Arquitectura multi-estrategia | Sistema puede cargar, testear y comparar N estrategias sin cambio de código | TBD | TBD |
| M3 | Paper trading robusto | La estrategia actual corre en paper trading 2 semanas sin intervención manual | TBD | TBD |
| M4 | Conexión a broker | Integración con broker elegido (Alpaca o IB) funcionando en paper trading | TBD | TBD |
| M5 | Primera semana real | Primera semana con dinero real, sin errores críticos, P&L registrado | TBD | TBD |

## Notas
- Editar “Milestone actual” cuando se cambie de foco.
- (Opcional) Agregar una lista de tickets por milestone para que el briefing pueda calcular X/Y.
