# QuantAgent-n96l / vbnh: punto de control de L16, criterio de la etapa 3 (2026-10-20)

Recálculo informativo (Q3 del plan) de las tres carteras que cortaron, con el criterio nuevo del DC `2026-10-19-DC-criterio-etapa3-y-loop-de-mejora` y el código de `lote/criterio-etapa3` en `5b809d1f` (E01 a E07 mergeados). Corrió el PM sobre `local-data`, con las curvas ya guardadas de las etapas 2 y 3. **No se corrió el motor ni se abrió ningún candado**: 2019–2022 ya estaba gastado para las tres (`hipotesis-candado.csv`). Solo agregados, sin precios. El paper trading es la vara fresca.

## Resultado

| Cartera | Sharpe etapa 3 | Benchmark etapa 3 | Sharpe etapa 2 | Benchmark etapa 2 | Años positivos | Drawdown | Veredicto |
|---|---:|---:|---:|---:|---:|---:|---|
| tendencia-momentum | 0,25 | 0,61 | 0,76 | 0,41 | 3 de 4 | 5,5 % | CORTA |
| transversal | 0,56 | 0,61 | 0,58 | 0,41 | 3 de 4 | 3,7 % | CORTA |
| estacionalidad | 0,18 | 0,57 | 0,45 | 0,39 | 3 de 4 | 34,3 % | CORTA |

| Cartera | DSR con N efectivo (95) | DSR con N crudo (99) | Recargo de paper si pasara |
|---|---:|---:|---|
| tendencia-momentum | 0,29 | 0,29 | 50 % del tamaño, 12 semanas |
| transversal | 0,53 | 0,52 | 75 %, 6 semanas |
| estacionalidad | 0,25 | 0,24 | 50 %, 12 semanas |

## Qué dice

- **Ninguna pasa.** El motivo es el mismo en las tres: el Sharpe de la etapa 3 es menor al del benchmark (misma cartera de activos, pesos iguales, rebalanceo mensual, sin señal). Con el criterio viejo cortaban por DSR; con el nuevo cortan por no superar a comprar y mantener.
- Cumplen Sharpe > 0, 3 de 4 años positivos y el límite de drawdown (estacionalidad con 34,3 % sobre 35 %).
- **Transversal es la más cercana**: 0,56 contra 0,61 y la única con DSR sobre 0,50. Quedó a 0,05 de pasar.
- Pasar de N crudo a N efectivo mueve el DSR en la segunda cifra decimal: la regla v1 solo quita repeticiones exactas.
- La etapa 2 de las tres sí supera al benchmark (0,76/0,45/0,58 contra 0,41/0,39/0,41), así que el corte aparece recién en 2019–2022.

## Cómo se comprobó

- El Sharpe de la etapa 3 y el de la etapa 2 recalculados desde las curvas coinciden con los registrados en `intentos.csv` (0,25/0,56/0,18 y 0,76/0,58/0,45).
- Sin filas sintéticas en el registro: 99 filas, 95 distintas.
- `python -m pytest -q -m oraculo`: ver el comentario del PR (resultado de la corrida del PM).
- Curvas usadas: `/tmp/claude-1000/pc55-salida`, `/tmp/corrida_transversal`, `/tmp/corrida_estacionalidad` (carpetas `equity` y `etapa3/equity`, no versionadas).

## Qué no se hizo

- No se corrió ninguna hipótesis nueva ni se leyó otra ventana que las de las etapas 2 y 3 ya abiertas.
- No se cambió el criterio ni el registro de intentos.
