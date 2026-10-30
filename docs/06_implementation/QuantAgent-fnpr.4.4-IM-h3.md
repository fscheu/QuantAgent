# QuantAgent-fnpr.4.4 — M5-33 H3: sector ETF → top-3 acciones

**Veredicto:** etapa 2 PASA (V30 sobre la curva de H3). Contra H1 el Sharpe no se distingue: ΔSharpe +0.03 ± 0.16 (error estándar bootstrap). Ventana 2011-2018, sin tocar la reserva 2023+.

## Qué es H3

H1 (rotación semanal de ETF sectoriales, máscara `sin-caida-m5`) con cada ETF elegido reemplazado por sus top-3 `candidatos_semanales` (partes iguales del peso del ETF). Si un sector no tiene candidatos queda el ETF; EEM/EWZ/SPY quedan como ETF. Semana en efectivo = efectivo. Regla: `cartera.pesos_h3`; clase `sector-score-h3` (alias de la métrica de H1); `hipotesis/m5-h3.yaml`.

## Resultados 2011-2018

| | H1 | H3 |
|---|---|---|
| Sharpe | 0.61 | 0.65 |
| CAGR | 6.9% | 9.1% |
| Max DD | 19.2% | 20.3% |
| Rotación (Σ\|Δw\| por año) | 22.1 | 33.9 |
| Años positivos | 6/8 | 7/8 |

Costo (slippage 0.05% por unidad rotada, igual para ambos): ~1.1%/año H1, ~1.7%/año H3.

- 418 semanas, 38 en efectivo. Acciones por semana: 6.87 (7.56 contando solo semanas invertidas), 282 distintas, 4102 trades.
- Respaldo ETF: 7 sector-semanas de 964 elegidas (7 semanas).

## Supervivencia (composición point-in-time)

Miembros-fecha de los sectores que H1 eligió (n = 54.056): elegible 69.3%, faltan_metricas 26.7%, sin_precio 3.1%, sin_cik 1.0%. Ninguna acción elegida quedó sin precio (0 de 282).

- Por año el elegible sube de 53% (2011) a 79% (2017); en 2018 sin_precio sube a 12% (el snapshot de precios termina ahí).
- Por sector: Consumer Staples 42% y Energy 51% quedan bajo 60% (mismo hueco que en `fnpr.4.10`); Real Estate 60%.
- Sesgo: el screener solo ve acciones con métricas y precio, así que sectores flacos pesan menos en acciones y más en el respaldo ETF.

## Decisiones

1. El embudo corre backtests por activo sobre un solo snapshot y no soporta acciones de otro snapshot con universo variable por fecha. La etapa 2 se resuelve con `scripts/m5/h3.py`, que arma la curva de H1 y H3 con el mismo código. No hay control del motor independiente ("sin control del motor").
2. Veredicto de etapa 2 = V30 sobre H3 (igual criterio que H2). H3 vs H1 se responde con ΔSharpe ± error.
3. Clase `sector-score-h3` sintética para pasar admisión (si no, "duplicada de m5-h1").
4. El costo se aplica a toda la rotación, en H1 y H3. Por eso el Sharpe de H1 acá (2011-2018) no es el del embudo (2007-2018).

## Reproducir

```
NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=.:scripts/m5 DATABASE_URL=sqlite:////tmp/ai-loop/h3.db \
QUANTAGENT_SNAPSHOT_DIR=local-data python scripts/m5/h3.py --cache h3-cache.pkl [--sin-registro]
```

Construir el cache tarda ~35 min; la clasificación de supervivencia ~20 min más.
