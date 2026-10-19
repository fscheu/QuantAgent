# QuantAgent-d4cq.68: punto de control del catálogo de primas (L13 / M3)

Punto de control de L13 ejecutado el 2026-10-19 sobre `local-data` con el snapshot `etf-1d-2026-10-amplio`. Se corrieron las carteras de primas de P08 (`cartera-transversal` y `cartera-estacionalidad`). Fede autorizó la apertura del período de prueba (2019–2022) para las carteras que pasaran la etapa 2.

## 1. Cartera transversal (`cartera-transversal`)

Dos piernas transversales sobre los ETFs del snapshot amplio, 0 parámetros libres: `momentum-transversal` (29 ETFs) y `baja-volatilidad` (17 ETFs de acciones), con pesos por igual riesgo y selección transversal mensual.

| Etapa | Período | Veredicto | Sharpe a tasa 0 | Trades | Ganancia (sobre 100.000) |
|---|---|---|---:|---:|---:|
| 0 admisión | | PASA | | | |
| 1 filtro, fuera de muestra | 2007–2018 | PASA (Sharpe de cartera 0,43) | 0,43 | 179 | |
| 2 control contra `backtesting.py` y V30 | 2007–2018 | PASA | 0,58 | 261 | 28.033,09 |
| 3 prueba, DSR | 2019–2022 | CORTA (DSR de cartera < 0,95) | 0,56 | 97 | 5.207,06 |

- Control de etapa 2: coincide en 46 de 46 celdas.
- DSR en etapa 3: 0,00 (requiere >= 0,95 frente a N = 105 intentos registrados en `intentos.csv`).
- Candado de prueba abierto y sellado el 2026-10-19T02:13:05 UTC.

## 2. Cartera de estacionalidad (`cartera-estacionalidad`)

Una pierna con la regla de Halloween sobre los 17 ETFs de acciones del snapshot amplio, 0 parámetros libres: largo de noviembre a abril, fuera de mayo a octubre.

| Etapa | Período | Veredicto | Sharpe a tasa 0 | Trades | Ganancia (sobre 100.000) |
|---|---|---|---:|---:|---:|
| 0 admisión | | PASA | | | |
| 1 filtro, fuera de muestra | 2007–2018 | PASA (Sharpe de cartera 0,36) | 0,36 | 136 | |
| 2 control contra `backtesting.py` y V30 | 2007–2018 | PASA | 0,45 | 204 | 81.906,56 |
| 3 prueba, DSR | 2019–2022 | CORTA (DSR de cartera < 0,95) | 0,18 | 68 | 6.360,40 |

- Control de etapa 2: coincide en 17 de 17 celdas.
- DSR en etapa 3: 0,00 (requiere >= 0,95 frente a N = 105 intentos registrados en `intentos.csv`).
- Candado de prueba abierto y sellado el 2026-10-19T02:25:16 UTC.

## 3. Estado en `catalogo/primas.yaml`

Las tres entradas correspondientes a L13 quedan actualizadas con su veredicto e informe:

| Prima id | Clase | Destino | Veredicto | Informe |
|---|---|---|---|---|
| `momentum-transversal` | `momentum-transversal` | L13 | CORTA | Este documento |
| `baja-volatilidad-transversal` | `baja-volatilidad` | L13 | CORTA | Este documento |
| `estacionalidad-halloween` | `halloween` | L13 | CORTA | Este documento |

## 4. Conclusiones y cierre de M3

- Ambas carteras superaron holgadamente el filtro (etapa 1) y el control con el motor propio (etapa 2), con 100% de coincidencia frente a `backtesting.py` (46/46 y 17/17).
- En la prueba 2019–2022, el decaimiento de rendimiento y la penalización estadística por la acumulación histórica de intentos (N = 105) sitúan el DSR por debajo del umbral de 0,95.
- Conforme a la regla D7, cada cartera suma exactamente una fila por etapa en `intentos.csv` (`activo = cartera`), protegiendo la varianza del DSR.
- Con este punto de control concluyen las evaluaciones de primas de L13 y se cierra el lote `lote/catalogo-de-primas` (hito M3).
