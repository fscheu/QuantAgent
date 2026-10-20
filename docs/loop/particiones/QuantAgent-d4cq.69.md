# QuantAgent-d4cq.69: partición de M01

El ticket M01 (`QuantAgent-d4cq.69`) abarca dos componentes complementarios:
1. Módulo puro `quantagent/fabrica/regimen.py` con `clasificar_regimen(cierres_spy)` (volatilidad 20d contra mediana de 252d y precio contra media de 200d, evaluados hasta t-1 con `shift(1)`), `sharpe_por_regimen(retornos, regimenes)` y `permutar_regimenes_bloques(regimenes, seed)` para contraste contra azar, junto con tests unitarios exhaustivos.
2. Script de medición sobre la cartera núcleo (`hipotesis/cartera-tendencia-momentum.yaml`) con datos reales 2007–2018 del snapshot amplio, ejecución de la prueba de permutación de bloques para contrastar contra ruido, y conclusión para decidir si la hipótesis se sigue o se archiva.

La implementación completa de ambas partes combinadas excede el objetivo de ≤100 líneas agregadas+borradas (estimada en ~180–230 líneas entre módulo puro, tests y script de medición), por lo que se divide en dos entregas acotadas conforme al PASO 3 de PLAN-CONTINUACION.md.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-d4cq.69.1 | Módulo puro `quantagent/fabrica/regimen.py`: `clasificar_regimen`, `sharpe_por_regimen` y `permutar_regimenes_bloques` | Test: clasificar_regimen en fecha t solo usa datos hasta t-1 (`shift(1)`); test: con series sintéticas conocidas, `sharpe_por_regimen` calcula el Sharpe diario por categoría; test: `permutar_regimenes_bloques` permuta bloques mensuales completos conservando la frecuencia de cada régimen; diff ≤100 líneas y CI verde | 60–80 | leer |
| QuantAgent-d4cq.69.2 | Medición de la cartera núcleo por régimen sobre 2007–2018 y contraste contra azar; depende de M01.1 | Script de medición produce la tabla de Sharpe diario de la cartera núcleo por régimen en 2007–2018; prueba de contraste reetiquetando regímenes al azar por bloques de un mes para ver si la diferencia supera el azar; conclusión de una línea (se sigue o se archiva); diff ≤100 líneas y CI verde | 60–80 | decidir |

Orden: .69.1 → .69.2. La cola de §4 de PLAN-CONTINUACION.md reemplaza la fila del padre por los dos hijos.
El padre queda `in_progress` hasta cerrar ambos hijos.
Cada hijo respeta el límite estricto de ≤100 líneas agregadas+borradas.
