# Paper M5: validación del sistema, no inversión

Fecha: 2026-10-31. Estado: **decidido por Fede el 2026-10-31**. No cambia código.
Relacionado: [`QuantAgent-fnpr.4.12-IM-etapa3`](../06_implementation/QuantAgent-fnpr.4.12-IM-etapa3.md), [`2026-10-19-DC-criterio-etapa3-y-loop-de-mejora`](./2026-10-19-DC-criterio-etapa3-y-loop-de-mejora.md), tickets `QuantAgent-fnpr.6` (L27) y `QuantAgent-fnpr.6.6` (M5-55).

## Resultado que motiva la decisión

- Etapa 2 (2011–2018): H1 (rotación sectorial con ETFs) y H3 (H1 con las 3 mejores acciones del screener por sector) pasan, con Sharpe 0,61 y 0,65. H2 y H4 cortan.
- Etapa 3 (2019–2022): cortan las dos. H1 da Sharpe 0,07 contra 0,60 del benchmark 1/N, con maxDD 37 %. H3 da 0,24 y corta porque tuvo 2 de 4 años positivos. ΔSharpe H3 − H1: +0,16 ± 0,21.
- Sesgo de supervivencia acotado, no eliminado: hasta 77 % de las semanas y hasta 56 % del peso pudieron estar afectados por acciones sin precio (68 de 671 miembros). Detalle en la IM de fnpr.4.12.

## Decisión

1. Lo de M5 se toma como **implementación inicial**. **No se pone dinero real** en estas estrategias, ni ahora ni como consecuencia de este paper.
2. El paper de M5 tiene un solo fin: **probar el sistema completo de punta a punta** (datos, señales, órdenes, stops, reconciliación, digest, monitoreo) y **armar los ciclos de aprendizaje y mejora** a partir del análisis de backtests y de paper.
3. Que H1 y H3 no hayan pasado la etapa 3 **no frena paper**: es lo ya previsto por R01 #4 (paper como prueba del sistema). Queda registrado como veredicto de etapa 3, sin reabrirlo.
4. Se da de alta **H3 en paper** junto con lo que ya corre (M5-55 / `fnpr.6.6`), como segunda pierna de la cuenta M5 (misma cuenta, base y bucket `m5`; USD 1.500 por pierna, órdenes etiquetadas por pierna), junto al pipeline ETF+arquetipos que ya arranca el 2026-11-02. H1 se compara en sombra (mismos datos, sin órdenes). El plan de paper en dos tiempos ya preveía sumar acciones al mismo pipeline cuando cerrara L28. El detalle de tickets está en Beads bajo `fnpr.6.6`.

## Consecuencias

- Los informes de paper no usan lenguaje de rendimiento para decidir capital: comparan **divergencia contra backtest**, costos y fallas del sistema.
- La confirmación con datos pagos (Sharadar, D1) deja de ser requisito previo de este paper; sigue siéndolo antes de cualquier decisión de capital real, que hoy no existe.
- Todo número de paper se lee con el sesgo de supervivencia de la IM de fnpr.4.12 a la vista.
- Cada ciclo de mejora se registra como intento nuevo en `intentos.csv` (regla de L9); paper no es holdout nuevo.
